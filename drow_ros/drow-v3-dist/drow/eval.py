import argparse
import glob
import matplotlib.pyplot as plt
import os
import pickle
import yaml

import utils.eval_utils as eu
from utils.dataset import create_test_dataloader


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=False, default=None,
                    help="specify the checkpoint to be evaluated")
parser.add_argument("--ckpt", type=str, required=False, default=None,
                    help="specify the checkpoint to be evaluated")
parser.add_argument("--pkl_dir", type=str, required=False, default=None,
                    help="specify the pkl file to be plotted")
parser.add_argument("--pkl", type=str, required=False, default=None,
                    help="specify the pkl file to be plotted")
parser.add_argument("--show", action='store_true', default=False, required=False)
parser.add_argument("--v2", action='store_true', default=False, required=False)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0]


def get_test_loader():
    test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                         num_scans=cfg['num_scans'],
                                         use_polar_grid=cfg['use_polar_grid'],
                                         cutout_kwargs=cfg['cutout_kwargs'],
                                         polar_grid_kwargs=cfg['polar_grid_kwargs'])
    return test_loader

def eval_ckpt_dir_v2(ckpt_dir):
    num_scans = 5
    cutout_kwargs = cfg['dataset_kwargs']['cutout_kwargs']
    cutout_kwargs['cutout_on_latest'] = True
    cutout_kwargs['center_on_latest'] = True
    test_loader = eu.create_test_dataloader(num_scans, cutout_kwargs)

    ckpt_file_list = glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    for ckpt_file in ckpt_file_list:
        model = eu.create_model(num_scans, ckpt_file, v2_format=True)
        results = eu.eval_model(model, test_loader, output_file=ckpt_file+".pkl")
        fig, ax = eu.plot_eval_result(results, plot_title=ckpt_file, output_file=ckpt_file+".pkl.png")
        if args.show:
            plt.show()


def eval_ckpt_v2(ckpt_file):
    num_scans = 5
    cutout_kwargs = cfg['dataset_kwargs']['cutout_kwargs']
    cutout_kwargs['cutout_on_latest'] = True
    cutout_kwargs['center_on_latest'] = True
    test_loader = eu.create_test_dataloader(num_scans, cutout_kwargs)

    model = eu.create_model(num_scans, ckpt_file, v2_format=True)
    results = eu.eval_model(model, test_loader, output_file=ckpt_file+".pkl")
    fig, ax = eu.plot_eval_result(results, plot_title=ckpt_file, output_file=ckpt_file+".pkl.png")
    if args.show:
        plt.show()


def eval_ckpt_dir(ckpt_dir):
    test_loader = get_test_loader()
    ckpt_file_list = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    for ckpt_file in ckpt_file_list:
        model = eu.create_model(cfg['num_scans'], ckpt_file, v2_format=False)
        results = eu.eval_model(model, test_loader, output_file=ckpt_file+".pkl")
        fig, ax = eu.plot_eval_result(results, plot_title=ckpt_file, output_file=ckpt_file+".pkl.png")
        if args.show:
            plt.show()


def eval_ckpt(ckpt_file):
    test_loader = get_test_loader()
    model = eu.create_model(cfg['num_scans'], ckpt_file, v2_format=False)
    results = eu.eval_model(model, test_loader, output_file=ckpt_file+".pkl")
    fig, ax = eu.plot_eval_result(results, plot_title=ckpt_file, output_file=ckpt_file+".pkl.png")
    if args.show:
        plt.show()


def plot_pkl_dir(pkl_dir):
    pkl_file_list = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    for pkl_file in pkl_file_list:
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        fig, ax = eu.plot_eval_result(results, plot_title=pkl_file, output_file=pkl_file+'.png')
        if args.show:
            plt.show()


def plot_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)
    fig, ax = eu.plot_eval_result(results, plot_title=pkl_file, output_file=pkl_file+'.png')
    if args.show:
        plt.show()


if __name__=='__main__':
    if args.v2:
        if args.ckpt_dir is not None:
            eval_ckpt_dir_v2(args.ckpt_dir)
        elif args.ckpt is not None:
            eval_ckpt_v2(args.ckpt)

    elif args.ckpt_dir is not None:
        eval_ckpt_dir(args.ckpt_dir)
    elif args.ckpt is not None:
        eval_ckpt(args.ckpt)
    elif args.pkl_dir is not None:
        plot_pkl_dir(args.pkl_dir)
    elif args.pkl is not None:
        plot_pkl(args.pkl)