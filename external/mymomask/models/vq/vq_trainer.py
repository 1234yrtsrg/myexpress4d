import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint import atomic_torch_save
from utils.distributed import is_main_process, reduce_mean, set_epoch_for_sampler, unwrap_model
from utils.utils import print_current_loss

def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        self.is_main = is_main_process(args)

        if args.is_train and self.is_main:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None
        self.recons_criterion = self._build_recons_criterion(args.recons_loss)
        self.latest_motions = None
        self.latest_pred_motion = None

    @staticmethod
    def _build_recons_criterion(loss_name):
        if loss_name == 'l1':
            return torch.nn.L1Loss()
        if loss_name == 'l1_smooth':
            return torch.nn.SmoothL1Loss()
        if loss_name in ['l2', 'mse']:
            return torch.nn.MSELoss()
        raise ValueError(f"Unsupported reconstruction loss: {loss_name}")

    def _extract_motion_tensor(self, batch_data):
        if torch.is_tensor(batch_data):
            return batch_data

        if isinstance(batch_data, np.ndarray):
            return torch.from_numpy(batch_data)

        if isinstance(batch_data, (list, tuple)):
            for item in batch_data:
                if torch.is_tensor(item) and item.ndim == 3:
                    return item
                if isinstance(item, np.ndarray) and item.ndim == 3:
                    return torch.from_numpy(item)

        raise TypeError('Expected batch_data to contain a motion tensor of shape [B, T, C].')

    def forward_step(self, batch_data):
        motions = self._extract_motion_tensor(batch_data).detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions)

        self.latest_motions = motions
        self.latest_pred_motion = pred_motion

        loss_recons = self.recons_criterion(pred_motion, motions)
        total_loss = loss_recons + self.opt.commit * loss_commit

        return {
            'loss': total_loss,
            'loss_recons': loss_recons,
            'loss_commit': loss_commit,
            'perplexity': perplexity,
        }

    def forward(self, batch_data):
        outputs = self.forward_step(batch_data)
        return (
            outputs['loss'],
            outputs['loss_recons'],
            outputs['loss_commit'],
            outputs['perplexity'],
        )


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        if not self.is_main:
            return
        state = {
            "vq_model": unwrap_model(self.vq_model).state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        atomic_torch_save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        unwrap_model(self.vq_model).load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, *unused_args, **unused_kwargs):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())
        best_val_loss = float('inf')

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            set_epoch_for_sampler(train_loader, epoch)
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss_dict = self.forward_step(batch_data)
                loss = loss_dict['loss']
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += reduce_mean(loss.item(), self.device)
                logs['loss_recons'] += reduce_mean(loss_dict['loss_recons'].item(), self.device)
                logs['loss_commit'] += reduce_mean(loss_dict['loss_commit'].item(), self.device)
                logs['perplexity'] += reduce_mean(loss_dict['perplexity'].item(), self.device)
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0 and self.logger is not None:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                elif it % self.opt.log_every == 0:
                    logs = defaultdict(def_value, OrderedDict())

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            val_loss = []
            val_loss_recons = []
            val_loss_commit = []
            val_perplexity = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss_dict = self.forward_step(batch_data)
                    val_loss.append(reduce_mean(loss_dict['loss'].item(), self.device))
                    val_loss_recons.append(reduce_mean(loss_dict['loss_recons'].item(), self.device))
                    val_loss_commit.append(reduce_mean(loss_dict['loss_commit'].item(), self.device))
                    val_perplexity.append(reduce_mean(loss_dict['perplexity'].item(), self.device))

            mean_val_loss = sum(val_loss) / len(val_loss)
            mean_val_loss_recons = sum(val_loss_recons) / len(val_loss_recons)
            mean_val_loss_commit = sum(val_loss_commit) / len(val_loss_commit)
            mean_val_perplexity = sum(val_perplexity) / len(val_perplexity)

            if self.logger is not None:
                self.logger.add_scalar('Val/loss', mean_val_loss, epoch)
                self.logger.add_scalar('Val/loss_recons', mean_val_loss_recons, epoch)
                self.logger.add_scalar('Val/loss_commit', mean_val_loss_commit, epoch)
                self.logger.add_scalar('Val/perplexity', mean_val_perplexity, epoch)

                print('Validation Loss: %.5f Reconstruction: %.5f, Commit: %.5f, Perplexity: %.5f' %
                      (mean_val_loss, mean_val_loss_recons, mean_val_loss_commit, mean_val_perplexity))

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                    print('Best validation model updated.')


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        total_iter = checkpoints.get('iter', checkpoints.get('niter', 0))
        return checkpoints['epoch'], total_iter

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        atomic_torch_save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss
