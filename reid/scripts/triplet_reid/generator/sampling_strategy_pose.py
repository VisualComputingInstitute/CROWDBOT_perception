from configs.datasets import *
from configs.evaluation import *
import json
import os


model_configs = {
    "single_head": {
        "name": "pose_reid",
        "backbone": None,
        "split": False,
        "single_head": True
    }

}
"""backbone_configs = {
    "groupnorm": {
        "name": "conv4_2_head_group",
        "ncpg": 16,
        "stride": 1,
        "init_from_file": "/home/pfeiffer/Projects/master-triplet-reid-pytorch/pretrained/resnet50_groupnorm16.tar"
    },
    "resnet50": {
        "name": "conv4_2_head_batch",
        "stride": 1
    }
}
"""

backbone_configs = {
    "groupnorm": {
        "name": "resnet_groupnorm", "ncpg": 16,
        "stride": 1,
        "init_from_file": "/home/pfeiffer/Projects/master-triplet-reid-pytorch/pretrained/resnet50_groupnorm16.tar"
    },
    "resnet50": {
        "name": "resnet",
        "stride": 1
    }
}


base_config = {
    "training": {
        "model": None,
        "sampler": {
            "name": None,
            "samplers": [{
                "name": "pk_sampler",
                "P": 18,
                "K": 4,
                "dataset": market_train_dataset
            },
            {
                "name": "random_sampler",
                "batch_size": 32,
                "dataset": mpii_train_dataset
            }]
        },
        "losses": {
            "type": "LinearWeightedLoss",
            "name": "Linear",
            "losses": [
                {
                    "name": "l2",
                    "type": "l2",
                    "endpoint": "pose",
                    "target": "coords",
                    "dataset": "mpii",
                    "weight": 1.0,
                    "log_sig_sq": 0.0
                },
                {
                    "name": "BatchHard",
                    "type": "BatchHard",
                    "endpoint" : "triplet",
                    "weight": 1.0,
                    "log_sig_sq": 0.0,
                    "margin": "soft",
                    "dataset": "market1501"
                }
            ],
        },

        "scheduler": {
            "preset": "huanghoujing",
            "epochs": 100,
            "t0": 70
        },
        "optimizer": {
            "name": "adam"
        },
        "checkpoint_frequency": 10
    },
    "validation": {
        "datasets": mpii_val_dataset
    },
    "evaluation": {
        "datasets": [duke_evaluation_dataset, market_evaluation_dataset],
        "delete": True
    }
}

experiment = 'sampling_strategy_pose_reid'

config_dir = os.path.join('configs', experiment)

if not os.path.isdir(config_dir):
    os.mkdir(config_dir)

log_dir = os.path.join("/globalwork/pfeiffer/master/", experiment)

# create logdir to avoid race condition when creating later
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

base_command = "python3 main.py with {} -n {} -c {} \
                -F {} -m veltins:27017:master"
sbash_command = "sbatch --partition=veltins,lopri -c 4 --gres=gpu:1 --mem 20G {}"

with open(os.path.join(config_dir, experiment + '.sh'), 'w') as f:
    f.write('#!/bin/bash\n')

    runidx = 0
    for sampler in ["RandomSamplerLongest", "SwitchingSamplerLongest", "RandomSamplerLongestKeep"]:
        for backbone_name, backbone_config in backbone_configs.items():

            config = base_config.copy()
            config["training"]["sampler"]["name"] = sampler
            model_config = model_configs['single_head'].copy()
            model_config['backbone'] = backbone_config
            config["training"]["model"] = model_config
            experiment_name = "{}_{}".format(sampler, backbone_name)
            path = os.path.join(config_dir, experiment_name)
            experiment_file = path + '.json'
            comment = experiment_name
            with open(experiment_file, 'w') as config_f:
                json.dump(config, config_f, indent=2)

            run_file = os.path.join(config_dir, 'run_{}.sh'.format(experiment_name))
            with open(run_file, 'w') as run_f:
                run_f.write('#!/bin/bash\n')
                run_f.write(base_command.format(experiment_file, experiment, experiment_name, log_dir) + '\n')
            f.write(sbash_command.format(run_file) + '\n')
            runidx += 1
