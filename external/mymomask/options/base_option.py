import argparse
import os
import torch

from utils.dataset_paths import configure_dataset_paths

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="blendshape_transformer", help='Name of this trial')

        self.parser.add_argument('--vq_name', type=str, default="rvq_blendshape", help='Name of the rvq model.')

        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')
        self.parser.add_argument("--num_gpus", type=int, default=1, help='Number of GPUs for a single training run')
        self.parser.add_argument("--master_port", type=int, default=29500, help='Master port used for distributed training')
        self.parser.add_argument("--num_workers", type=int, default=4, help='Number of dataloader workers per process')
        self.parser.add_argument('--dataset_name', type=str, default='blendshape', help='Dataset identifier')
        self.parser.add_argument('--data_root', type=str, default='./dataset/blendshape', help='Dataset root directory')
        self.parser.add_argument('--motion_dir', type=str, default='', help='Directory containing motion numpy files')
        self.parser.add_argument('--text_dir', type=str, default='', help='Directory containing paired text files')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here.')

        self.parser.add_argument('--latent_dim', type=int, default=384, help='Dimension of transformer latent.')
        self.parser.add_argument('--n_heads', type=int, default=6, help='Number of heads.')
        self.parser.add_argument('--n_layers', type=int, default=8, help='Number of attention layers.')
        self.parser.add_argument('--ff_size', type=int, default=1024, help='FF_Size')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio in transformer')

        self.parser.add_argument("--max_motion_length", type=int, default=196, help="Max sequence length in frames")
        self.parser.add_argument("--min_motion_length", type=int, default=16, help="Min sequence length in frames")
        self.parser.add_argument("--unit_length", type=int, default=4, help="Downscale ratio of VQ")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Maximum number of text tokens before adding special tokens")

        self.parser.add_argument('--force_mask', action="store_true", help='True: mask out conditions')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train
        self.opt = configure_dataset_paths(self.opt)

        if self.opt.gpu_id != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
