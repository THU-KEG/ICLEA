import os

import torch

from model.ICLEA import Trainer


if __name__ == '__main__':
    print('PyTorch:', torch.__version__)
    print('CUDA:', torch.version.cuda)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>'))
    print('Visible CUDA devices:', torch.cuda.device_count())
    trainer = Trainer()
    trainer.train(0)
