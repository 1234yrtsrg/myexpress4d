import argparse
import os

import torch

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.motion_loaders.vqvae_model_dataset import ExternalMoMaskGenerator, get_vqvae_loader
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from diffusion import logger
from eval.eval_humanml import evaluation
from utils import dist_util
from utils.fixseed import fixseed
from utils.parser_util import add_base_options, add_classifier_options, add_data_options, add_evaluation_options, add_flame_options

torch.multiprocessing.set_sharing_strategy("file_system")


def evaluation_vae_parser():
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_evaluation_options(parser)
    add_flame_options(parser)
    add_classifier_options(parser)

    group = parser.add_argument_group("vqvae")
    group.add_argument("--vq_model_root", default="/output_model/vq-vae", type=str,
                       help="Directory containing rvq_Express4D, t2m_Express4D, res_Express4D.")
    group.add_argument("--vq_code_root", default="./external/mymomask", type=str,
                       help="External VQ-VAE/MoMask source root.")
    group.add_argument("--rvq_checkpoint", default="finest.tar", type=str,
                       help="Checkpoint under rvq_Express4D/model.")
    group.add_argument("--t2m_checkpoint", default="net_best_acc.tar", type=str,
                       help="Checkpoint under t2m_Express4D/model.")
    group.add_argument("--res_checkpoint", default="net_best_loss.tar", type=str,
                       help="Checkpoint under res_Express4D/model.")
    group.add_argument("--no_residual", action="store_true",
                       help="Disable residual transformer refinement.")
    group.add_argument("--time_steps", default=18, type=int,
                       help="Mask transformer sampling steps.")
    group.add_argument("--cond_scale", default=4.0, type=float,
                       help="Classifier-free guidance scale for the text-to-motion transformer.")
    group.add_argument("--res_cond_scale", default=5.0, type=float,
                       help="Classifier-free guidance scale for the residual transformer.")
    group.add_argument("--temperature", default=1.0, type=float,
                       help="Sampling temperature.")
    group.add_argument("--topk_filter_thres", default=0.9, type=float,
                       help="Top-k filter threshold used by MoMask-style sampling.")
    group.add_argument("--gumbel_sample", action="store_true",
                       help="Use gumbel sampling if the external transformer supports it.")
    return parser.parse_args()


if __name__ == "__main__":
    args = evaluation_vae_parser()
    fixseed(args.seed)
    args.batch_size = 32

    eval_dataset_name = args.eval_dataset_override if args.eval_dataset_override else args.dataset
    model_root_name = os.path.basename(os.path.normpath(args.vq_model_root))
    log_dir = os.path.join(args.vq_model_root, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eval_humanml_vae_{model_root_name}_{args.eval_mode}_seed{args.seed}.log")

    print(f"Will save to log file [{log_file}]")
    print(f"Eval mode [{args.eval_mode}]")
    if args.eval_mode == "debug":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 48
        replication_times = 20
    elif args.eval_mode == "wo_mm":
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 48
        replication_times = 3
    elif args.eval_mode == "mm_short":
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 48
        replication_times = 10
    else:
        raise ValueError()

    if args.eval_rep_times > 0:
        replication_times = args.eval_rep_times

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = getattr(args, "eval_split", "test")
    gen_loader = get_dataset_loader(
        name=eval_dataset_name,
        batch_size=args.batch_size,
        num_frames=None,
        data_mode=args.data_mode,
        max_len=args.maximum_frames,
        flip_face_on=False,
        fps=args.fps,
        split=split,
        hml_mode="generator",
    )
    gt_loader = get_dataset_loader(
        name=eval_dataset_name,
        batch_size=args.batch_size,
        num_frames=None,
        data_mode=args.data_mode,
        max_len=args.maximum_frames,
        flip_face_on=False,
        fps=args.fps,
        split=split,
        hml_mode="gt",
    )

    logger.log("Creating VQ-VAE/T2M generator...")
    generator = ExternalMoMaskGenerator(
        model_root=args.vq_model_root,
        code_root=args.vq_code_root or args.vq_model_root,
        device=dist_util.dev(),
        use_ema=args.use_ema,
        use_residual=not args.no_residual,
        t2m_checkpoint=args.t2m_checkpoint,
        rvq_checkpoint=args.rvq_checkpoint,
        res_checkpoint=args.res_checkpoint,
        time_steps=args.time_steps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topk_filter_thres=args.topk_filter_thres,
        gumbel_sample=args.gumbel_sample,
        res_cond_scale=args.res_cond_scale,
    )

    eval_motion_loaders = {
        "vald": lambda: get_vqvae_loader(
            generator,
            args.batch_size,
            gen_loader,
            mm_num_samples,
            mm_num_repeats,
            gt_loader.dataset.opt.max_motion_length,
            num_samples_limit,
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(eval_dataset_name, args.eval_model_name, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
