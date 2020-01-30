import argparse
import os

import yaml
import wandb

import torch
from torch import optim

from model.drow import FastDROWNet3LF2p, model_fn, model_fn_eval
from utils.dataset import create_dataloader, create_test_dataloader
from utils.logger import create_logger, create_tb_logger
from utils.train_utils import Trainer, LucasScheduler
import utils.eval_utils as eu


torch.backends.cudnn.benchmark = True  # Run benchmark to select fastest implementation of ops.

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0]

wandb.init(project="d-v3", name=cfg['name'], sync_tensorboard=True)
wandb.config.update(cfg)


if __name__=='__main__':
    root_result_dir = os.path.join('./', 'output', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, tb_logger = create_logger(root_result_dir), create_tb_logger(root_result_dir)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # create dataloader & network & optimizer
    train_loader, eval_loader = create_dataloader(data_path="../data/DROWv2-data",
                                                  num_scans=cfg['num_scans'],
                                                  batch_size=cfg['batch_size'],
                                                  num_workers=cfg['num_workers'],
                                                  use_polar_grid=cfg['use_polar_grid'],
                                                  train_with_val=cfg['train_with_val'],
                                                  use_data_augumentation=cfg['use_data_augumentation'],
                                                  cutout_kwargs=cfg['cutout_kwargs'],
                                                  polar_grid_kwargs=cfg['polar_grid_kwargs'])

    # model = DROWNet3LF2p()
    model = FastDROWNet3LF2p(num_scans=cfg['num_scans'])
    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    lr_scheduler = LucasScheduler(optimizer, 30, 1e-3, 40, 1e-6)
    model.cuda()

    wandb.watch(model, log="all")

    # start training
    logger.info('**********************Start training**********************')

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        model_fn_eval=model_fn_eval,
        tb_log=tb_logger,
        grad_norm_clip=cfg['grad_norm_clip'],
        logger=logger)

    trainer.train(num_epochs=cfg['epochs'],
                  train_loader=train_loader,
                  eval_loader=eval_loader,
                  eval_frequency=5,
                  ckpt_save_interval=5,
                  lr_scheduler_each_iter=True)

    # testing
    logger.info('**********************Start testing**********************')
    test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         use_polar_grid=cfg['use_polar_grid'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'])
    eval_rpt = eu.eval_model(model, test_loader,
                             output_file=os.path.join(root_result_dir, 'eval_rpt.pkl'))
    eu.plot_eval_result(eval_rpt,
                        plot_title='%s, %s epoch' % (cfg['name'], cfg['epochs']),
                        output_file=os.path.join(root_result_dir, 'eval_rpt.png'))
    logger.info('**********************End**********************')
