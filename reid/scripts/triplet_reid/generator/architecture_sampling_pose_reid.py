from configs.datasets import *
from configs.evaluation import *
import json
import os


model_configs = {
    "two_head_group": {
        "name": "pose_reid",
        "backbone": {
            "name": "conv4_2_head_group",
            "ncpg": 16,
            "stride": 1,
            "init_from_file": "/home/pfeiffer/Projects/master-triplet-reid-pytorch/pretrained/resnet50_groupnorm16.tar"
        },
        "split": False,
        "single_head": False
    },
}


weightings = {
        "1:1": [1, 1],
        "3:1": [3, 1],
        "equal": "equal"
}


schedulers = {
        "long": (300, 151),
        "short": (100, 70)
}


num_workers = 6

base_config = {
    "training": {
        "model": None,
        "sampler": {
            "type": "random_sampler_length_weighted",
            "samplers": {
                "market_sampler": {
                    "type": "pk_sampler",
                    "P": 18,
                    "K": 4,
                    "dataset": market_train_dataset
                },
                "mpii_sampler": {
                    "type": "random_sampler",
                    "batch_size": 32,
                    "dataset": mpii_train_dataset
                }
            },
            "weights": None
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
            "epochs": None,
            "t0": None
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
    },
    "num_workers": num_workers
}


experiment = 'architecture_sampling_pose_reid_v2'

config_dir = os.path.join('configs', experiment)

if not os.path.isdir(config_dir):
    os.mkdir(config_dir)

log_dir = os.path.join("/globalwork/pfeiffer/master/", experiment)

# create logdir to avoid race condition when creating later
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

base_command = "python3 main.py with {} -n {} -c {} \
                -F {} -m veltins:27017:master"


SLURM_SETTINGS = """#SBATCH --partition=lopri
#SBATCH --signal=TERM@120
#SBATCH --time=8-00:00:00
#SBATCH --cpus-per-task={num_workers}
#SBATCH --gres=gpu:1
#SBATCH --mem=33G\n"""

SHEBANG = "#!/bin/bash\n"

class ConcurrentManager(object):
    SLURM_SETTINGS = SLURM_SETTINGS
    BASE_PATH = os.path.join(config_dir, "runner_{}.sh")
    def __init__(self, num_concurrent):
        self.num_concurrent = num_concurrent
        self.idx = 0
        self.files = [None] * num_concurrent

    def get_file(self):
        self.idx = self.idx % self.num_concurrent
        if self.files[self.idx] is None:
            f = open(self.BASE_PATH.format(self.idx), 'w')
            def write_header(f):
                f.write(SHEBANG)
                f.write(self.SLURM_SETTINGS)
            write_header(f)
            self.files[self.idx] = f

        f = self.files[self.idx]
        self.idx += 1
        return f

    def add_job(self, command):
        f = self.get_file()
        f.write("srun " + command + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for file in self.files:
            if file is not None:
                file.close()

TEST_COMMAND = "python3 main.py test_config with {} -u"

with ConcurrentManager(num_concurrent=1) as manager:
    with open(os.path.join(config_dir, 'test.sh'), 'w') as test_file:
        test_file.write(SHEBANG)
        for model_name, model_config in model_configs.items():
            for weight_name, weighting in weightings.items():
                for scheduler_name, scheduler in schedulers.items():
                    config = base_config.copy()
                    config["training"]["model"] = model_config
                    config["training"]["scheduler"]["epochs"] = scheduler[0]
                    config["training"]["scheduler"]["t0"] = scheduler[1]
                    config["training"]["sampler"]["weights"] = weighting

                    experiment_name = "{}_{}_{}".format(model_name, scheduler_name, weight_name)
                    path = os.path.join(config_dir, experiment_name)
                    experiment_file = path + '.json'
                    comment = experiment_name
                    with open(experiment_file, 'w') as config_f:
                        json.dump(config, config_f, indent=2)
                    command = base_command.format(experiment_file, experiment, experiment_name, log_dir)
                    manager.add_job(command)
                    test_file.write(TEST_COMMAND.format(experiment_file) + '\n')
