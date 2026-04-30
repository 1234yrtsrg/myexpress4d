import importlib
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def _as_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _strip_state_dict_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _state_dict_from_checkpoint(checkpoint, preferred_keys):
    for key in preferred_keys:
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key], key

    if isinstance(checkpoint, dict):
        tensor_like = [torch.is_tensor(value) for value in checkpoint.values()]
        if tensor_like and all(tensor_like):
            return checkpoint, "<root>"

    raise KeyError(
        "Could not find a loadable state_dict in checkpoint. "
        f"Tried keys: {', '.join(preferred_keys)}"
    )


def _load_checkpoint_into(module, checkpoint_path, normal_keys, use_ema):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ema_keys = [
        "model_ema",
        "ema",
        "net_ema",
        "model_avg",
        "vq_model_ema",
        "t2m_transformer_ema",
        "res_transformer_ema",
    ]
    keys = (ema_keys + normal_keys) if use_ema else normal_keys
    state_dict, loaded_key = _state_dict_from_checkpoint(checkpoint, keys)
    state_dict = _strip_state_dict_prefixes(state_dict)
    matched_keys = set(state_dict.keys()).intersection(module.state_dict().keys())
    if not matched_keys:
        raise RuntimeError(
            f"Checkpoint [{checkpoint_path}] key [{loaded_key}] did not match any parameters in "
            f"{module.__class__.__name__}. Check the model code root and checkpoint type."
        )
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading {checkpoint_path}: {len(unexpected)}")
    print(f"Loaded [{checkpoint_path}] using checkpoint key [{loaded_key}]")
    return checkpoint


class ExternalMoMaskGenerator:
    """Runtime adapter for MoMask-style RVQ + text transformer checkpoints."""

    def __init__(
        self,
        model_root,
        code_root,
        device,
        use_ema=True,
        use_residual=True,
        t2m_checkpoint="net_best_acc.tar",
        rvq_checkpoint="finest.tar",
        res_checkpoint="net_best_loss.tar",
        unit_length=4,
        time_steps=18,
        cond_scale=4.0,
        temperature=1.0,
        topk_filter_thres=0.9,
        gumbel_sample=False,
        res_cond_scale=5.0,
    ):
        self.model_root = _as_path(model_root)
        self.code_root = _as_path(code_root or model_root)
        self.device = device
        self.use_ema = use_ema
        self.use_residual = use_residual
        self.t2m_checkpoint = t2m_checkpoint
        self.rvq_checkpoint = rvq_checkpoint
        self.res_checkpoint = res_checkpoint
        self.unit_length = unit_length
        self.time_steps = time_steps
        self.cond_scale = cond_scale
        self.temperature = temperature
        self.topk_filter_thres = topk_filter_thres
        self.gumbel_sample = gumbel_sample
        self.res_cond_scale = res_cond_scale

        self._prepare_external_imports()
        self._build_models()

    def _prepare_external_imports(self):
        if not os.path.isdir(self.model_root):
            raise FileNotFoundError(
                f"VQ-VAE model root was not found: {self.model_root}. "
                "Set --vq_model_root to the directory containing rvq_Express4D/t2m_Express4D/res_Express4D."
            )
        if not os.path.isdir(self.code_root):
            raise FileNotFoundError(
                f"VQ-VAE code root was not found: {self.code_root}. "
                "Set --vq_code_root to the external VQ-VAE/MoMask source directory."
            )

        sys.path.insert(0, self.code_root)
        # The external project commonly has top-level packages named utils/models/options.
        # Remove already-imported local packages with those names so external imports resolve correctly.
        for package_name in ("models", "options"):
            sys.modules.pop(package_name, None)
        if "utils.get_opt" not in sys.modules:
            sys.modules.pop("utils", None)

        try:
            self.get_opt = importlib.import_module("utils.get_opt").get_opt
            self.RVQVAE = importlib.import_module("models.vq.model").RVQVAE
            transformer_module = importlib.import_module("models.mask_transformer.transformer")
            self.MaskTransformer = transformer_module.MaskTransformer
            self.ResidualTransformer = getattr(transformer_module, "ResidualTransformer", None)
        except Exception as exc:
            raise ImportError(
                "Could not import the external VQ-VAE/MoMask model code. Expected modules like "
                "utils.get_opt, models.vq.model, and models.mask_transformer.transformer under "
                f"[{self.code_root}]."
            ) from exc

    def _opt(self, exp_name):
        return self.get_opt(pjoin(self.model_root, exp_name, "opt.txt"), self.device)

    def _build_models(self):
        self.vq_opt = self._opt("rvq_Express4D")
        self.t2m_opt = self._opt("t2m_Express4D")
        self.t2m_opt.num_tokens = self.vq_opt.nb_code
        self.t2m_opt.num_quantizers = self.vq_opt.num_quantizers
        self.t2m_opt.code_dim = self.vq_opt.code_dim

        self.unit_length = int(getattr(self.vq_opt, "unit_length", self.unit_length))
        clip_version = getattr(self.t2m_opt, "clip_version", "ViT-B/32")
        input_width = getattr(self.vq_opt, "input_width", self.vq_opt.dim_pose)

        self.vq_model = self.RVQVAE(
            self.vq_opt,
            input_width,
            self.vq_opt.nb_code,
            self.vq_opt.code_dim,
            self.vq_opt.output_emb_width,
            self.vq_opt.down_t,
            self.vq_opt.stride_t,
            self.vq_opt.width,
            self.vq_opt.depth,
            self.vq_opt.dilation_growth_rate,
            self.vq_opt.vq_act,
            self.vq_opt.vq_norm,
        )
        _load_checkpoint_into(
            self.vq_model,
            pjoin(self.model_root, "rvq_Express4D", "model", self.rvq_checkpoint),
            normal_keys=["net", "vq_model", "model", "state_dict"],
            use_ema=self.use_ema,
        )
        self.vq_model.to(self.device)
        self.vq_model.eval()

        self.t2m_transformer = self.MaskTransformer(
            code_dim=self.vq_opt.code_dim,
            cond_mode="text",
            latent_dim=self.t2m_opt.latent_dim,
            ff_size=self.t2m_opt.ff_size,
            num_layers=self.t2m_opt.n_layers,
            num_heads=self.t2m_opt.n_heads,
            dropout=self.t2m_opt.dropout,
            clip_dim=512,
            cond_drop_prob=self.t2m_opt.cond_drop_prob,
            clip_version=clip_version,
            opt=self.t2m_opt,
        )
        _load_checkpoint_into(
            self.t2m_transformer,
            pjoin(self.model_root, "t2m_Express4D", "model", self.t2m_checkpoint),
            normal_keys=["t2m_transformer", "trans", "transformer", "net", "model", "state_dict"],
            use_ema=self.use_ema,
        )
        self.t2m_transformer.to(self.device)
        self.t2m_transformer.eval()

        self.res_transformer = None
        res_dir = pjoin(self.model_root, "res_Express4D")
        res_ckpt = pjoin(res_dir, "model", self.res_checkpoint)
        if self.use_residual and self.ResidualTransformer is not None and os.path.exists(res_ckpt):
            self.res_opt = self._opt("res_Express4D")
            self.res_opt.num_quantizers = self.vq_opt.num_quantizers
            self.res_opt.num_tokens = self.vq_opt.nb_code
            self.res_opt.code_dim = self.vq_opt.code_dim
            self.res_transformer = self.ResidualTransformer(
                code_dim=self.vq_opt.code_dim,
                cond_mode="text",
                latent_dim=self.res_opt.latent_dim,
                ff_size=self.res_opt.ff_size,
                num_layers=self.res_opt.n_layers,
                num_heads=self.res_opt.n_heads,
                dropout=self.res_opt.dropout,
                clip_dim=512,
                shared_codebook=self.vq_opt.shared_codebook,
                cond_drop_prob=self.res_opt.cond_drop_prob,
                share_weight=getattr(self.res_opt, "share_weight", False),
                clip_version=getattr(self.res_opt, "clip_version", clip_version),
                opt=self.res_opt,
            )
            _load_checkpoint_into(
                self.res_transformer,
                res_ckpt,
                normal_keys=["res_transformer", "transformer", "net", "model", "state_dict"],
                use_ema=self.use_ema,
            )
            self.res_transformer.to(self.device)
            self.res_transformer.eval()

    def _generate_tokens(self, captions, token_lens):
        try:
            return self.t2m_transformer.generate(
                captions,
                token_lens,
                timesteps=self.time_steps,
                cond_scale=self.cond_scale,
                temperature=self.temperature,
                topk_filter_thres=self.topk_filter_thres,
                gsample=self.gumbel_sample,
            )
        except TypeError:
            return self.t2m_transformer.generate(
                captions,
                token_lens,
                self.time_steps,
                self.cond_scale,
                self.temperature,
                self.topk_filter_thres,
                self.gumbel_sample,
            )

    def _apply_residual(self, token_ids, captions, token_lens):
        if self.res_transformer is None:
            return token_ids
        try:
            return self.res_transformer.generate(
                token_ids,
                captions,
                token_lens,
                temperature=self.temperature,
                topk_filter_thres=self.topk_filter_thres,
                cond_scale=self.res_cond_scale,
            )
        except TypeError:
            return self.res_transformer.generate(
                token_ids,
                captions,
                token_lens,
                temperature=self.temperature,
                cond_scale=self.res_cond_scale,
            )

    def generate(self, captions, lengths):
        lengths = torch.as_tensor(lengths, device=self.device).long()
        token_lens = torch.clamp(lengths // self.unit_length, min=1)

        with torch.no_grad():
            token_ids = self._generate_tokens(captions, token_lens)
            token_ids = self._apply_residual(token_ids, captions, token_lens)
            motions = self.vq_model.forward_decoder(token_ids)

        if isinstance(motions, tuple):
            motions = motions[0]
        return motions.detach().float()


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data["mm_motions"]
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion["length"])
            motions.append(mm_motion["motion"][None, :])
        m_lens = np.array(m_lens, dtype=np.int64)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        return motions[sort_indx], m_lens[sort_indx]


class CompVQVAEGeneratedDataset(Dataset):
    def __init__(
        self,
        generator,
        dataloader,
        mm_num_samples,
        mm_num_repeats,
        max_motion_length,
        num_samples_limit,
    ):
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        self.max_motion_length = max_motion_length

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print("real_num_batches", real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size + 1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print("mm_idxs", mm_idxs)

        with torch.no_grad():
            for i, (_, model_kwargs) in tqdm(enumerate(dataloader)):
                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                captions = model_kwargs["y"]["text"]
                lengths = model_kwargs["y"]["lengths"].cpu().numpy().astype(np.int64)
                tokens = [token_string.split("_") for token_string in model_kwargs["y"]["tokens"]]

                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for repeat_idx in range(repeat_times):
                    sample = generator.generate(captions, lengths)
                    sample = self._normalize_sample_shape(sample)

                    if repeat_idx == 0:
                        sub_dicts = []
                        for bs_i in range(dataloader.batch_size):
                            motion, motion_length = self._pad_or_trim(sample[bs_i].cpu().numpy(), lengths[bs_i])
                            sub_dicts.append({
                                "motion": motion,
                                "length": motion_length,
                                "caption": captions[bs_i],
                                "tokens": tokens[bs_i],
                                "cap_len": tokens[bs_i].index("eos/OTHER") + 1
                                if "eos/OTHER" in tokens[bs_i] else len(tokens[bs_i]),
                            })
                        generated_motion += sub_dicts

                    if is_mm:
                        for bs_i in range(dataloader.batch_size):
                            motion, motion_length = self._pad_or_trim(sample[bs_i].cpu().numpy(), lengths[bs_i])
                            mm_motions.append({
                                "motion": motion,
                                "length": motion_length,
                            })

                if is_mm:
                    mm_generated_motions += [
                        {
                            "caption": captions[bs_i],
                            "tokens": tokens[bs_i],
                            "cap_len": tokens[bs_i].index("eos/OTHER") + 1
                            if "eos/OTHER" in tokens[bs_i] else len(tokens[bs_i]),
                            "mm_motions": mm_motions[bs_i::dataloader.batch_size],
                        }
                        for bs_i in range(dataloader.batch_size)
                    ]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def _normalize_sample_shape(self, sample):
        if sample.dim() != 3:
            raise ValueError(f"Expected generated motion with shape [B, T, D] or [B, D, T], got {tuple(sample.shape)}")
        feature_dim = getattr(self.dataset, "mean", np.zeros(61)).shape[-1]
        if sample.shape[-1] == feature_dim:
            return sample
        if sample.shape[1] == feature_dim:
            return sample.permute(0, 2, 1)
        raise ValueError(f"Generated motion shape {tuple(sample.shape)} does not match feature dim {feature_dim}")

    def _pad_or_trim(self, motion, length):
        length = int(min(length, self.max_motion_length, motion.shape[0]))
        feature_dim = motion.shape[-1]
        padded = np.zeros((self.max_motion_length, feature_dim), dtype=np.float32)
        padded[:length] = motion[:length].astype(np.float32)
        return padded, length

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data["motion"], data["length"], data["caption"], data["tokens"]
        sent_len = data["cap_len"]

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens)


def get_vqvae_loader(
    generator,
    batch_size,
    ground_truth_loader,
    mm_num_samples,
    mm_num_repeats,
    max_motion_length,
    num_samples_limit,
):
    opt = {"name": "vqvae"}
    print("Generating %s ..." % opt["name"])
    dataset = CompVQVAEGeneratedDataset(
        generator,
        ground_truth_loader,
        mm_num_samples,
        mm_num_repeats,
        max_motion_length,
        num_samples_limit,
    )
    mm_dataset = MMGeneratedDataset(opt, dataset, ground_truth_loader.dataset.w_vectorizer)
    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)
    print("Generated Dataset Loading Completed!!!")
    return motion_loader, mm_motion_loader
