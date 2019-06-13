from configs.datasets import *
from configs.evaluation import *
import json
import os


model_configs = {
    "groupnorm": {
        "name": "baseline",
        "backbone": {
            "name": "resnet_groupnorm",
            "ncpg": 16,
            "stride": 1,
            "init_from_file": "/home/pfeiffer/Projects/master-triplet-reid-pytorch/pretrained/resnet50_groupnorm16.tar"
        },
        "pooling": "max"
    },
    "resnet50": {
        "name": "baseline",
        "backbone": {
            "name": "resnet",
            "stride": 1
        },
        "pooling": "max"
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
                "name": "pk_sampler",
                "P": 18,
                "K": 4,
                "dataset": duke_train_dataset
            }]
        },
        "losses": {
            "name": "BatchHard",
            "type": "BatchHard",
            "endpoint" : "triplet",
            "weight": 1.0,
            "dataset": "all"
        },
        "scheduler": {
            "preset": "huanghoujing"
        },
        "optimizer": {
            "name": "adam"
        }
    },
    "evaluation": {
        "datasets": [duke_evaluation_dataset, market_evaluation_dataset],
        "delete": True
    }
}

experiment = 'sampling_strategy'

config_dir = os.path.join('configs', experiment)
if not os.path.isdir(config_dir):
    os.mkdir(config_dir)


log_dir = os.path.join("/globalwork/pfeiffer/master/", experiment)

# create logdir to avoid race condition when creating later
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

base_command = "python3 main.py with {} -n sampling_strategy -c {} \
                -F {} -m veltins:27017:master"
sbash_command = "sbatch --partition=veltins -c 4 --gres=gpu:1 --mem 20G {}"

with open(os.path.join(config_dir, experiment + '.sh'), 'w') as f:
    f.write('#!/bin/bash\n')

    runidx = 0
    for sampler in ["SwitchingSamplerShortest", "SwitchingSamplerLongest", "RandomSamplerLongestKeep", "RandomSamplerLongest"]:
        for model_name, model_config in model_configs.items():
            config = base_config.copy()
            config["training"]["sampler"]["name"] = sampler
            config["training"]["model"] = model_config
            experiment_name = "{}_{}".format(sampler, model_name)
            path = os.path.join(config_dir, experiment_name)
            experiment_file = path + '.json'
            comment = experiment_name
            with open(experiment_file, 'w') as config_f:
                json.dump(config, config_f, indent=2)

            run_file = os.path.join(config_dir, 'run_{}.sh'.format(experiment_name))
            with open(run_file, 'w') as run_f:
                run_f.write('#!/bin/bash\n')
                run_f.write(base_command.format(experiment_file, experiment_name, log_dir) + '\n')
            f.write(sbash_command.format(run_file) + '\n')
            runidx += 1
