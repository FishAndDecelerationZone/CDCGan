


import os
import datetime

from CGan_DC.CGan import CGan
from CGan_DC.MnistGenerator import MnistGenerator
from CGan_DC.Config import Config


def run_main():
    """
    这是主函数
    """
    batch_size = 512
    cfg = Config()
    cgan = CGan(cfg)
    train_datagener = MnistGenerator(batch_size, cfg.generator_noise_input_dim)
    cgan.train(train_datagener,epoch=10000, k=5,batch_size=batch_size)


if __name__ == '__main__':
    run_main()
