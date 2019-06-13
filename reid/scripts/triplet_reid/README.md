# Description
A pytorch implementation of the "In Defense of the Triplet Loss for Person Re-Identification" paper (https://arxiv.org/abs/1703.07737).

This repository also contains also a implementation of the MGN network from the paper: 
["Learning Discriminative Features with Multiple Granularities for Person Re-Identification"](https://arxiv.org/abs/1804.01438)

It is built from scratch with ease in experimenting, modification and reproducability.
However, it still work in progress.

# Requirements

- numpy
- Pillow
- h5py
- scipy
- torch
- torchvision
- sacred
- imgaug

This repository depends for evaluation on the original repository:
https://github.com/VisualComputingInstitute/triplet-reid

For evaluation:
- tensorflow
- scikit-learn
- scipy
- h5py

# Installation
git clone --recurse-submodules https://github.com/kilsenp/triplet-reid-pytorch.git
pip install -r requirements.txt (for training)

To be able to use omniboard:
- Install mongodb
- Install npm
- Install omniboard

### For no-admin rights user:
#### mongodb
Install tarball from https://docs.mongodb.com/v3.2/tutorial/install-mongodb-on-linux/

#### npm
I recommend setting npm up in such way that installed modules can be run from command line.

Therefore:
- Change the npm install directory:
-- create a file called .npmrc with the content
`prefix=${HOME}/.npm-packages`
-- add the following to your .bashrc
`# Node setup for global packages without sudo
NPM_PACKAGES="${HOME}/.npm-packages"
NODE_PATH="$NPM_PACKAGES/lib/node_modules:$NODE_PATH"
PATH="$NPM_PACKAGES/bin:$PATH"`


#### omniboard
`npm install omniboard`

# Starting 
mongod --dbpath mongo
## Access database from remote host
- Create a config file with:
`bind_ip = 127.0.0.1, ip1, ip2`
where ip1 and ip2 are the assigned apis within the network the database should be accessible from.

Then start mongodb with
mongod --dbpath mongo -f mongod.conf

omniboard -m host:port:db
### Password Protect Database
- Create admin user.
```
use admin
db.createUser(
  {
    user: "myUserAdmin",
    pwd: "abc123",
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
```
- restart database with `mongod --auth` + additional parameters
#### Additionally
This is not really necessary but for one good practice but also seems to be necessary to work 
correctly with omniboard
- Create another user in your experiment database
```
use master
db.createUser(
  {
    user: "myUser",
    pwd: "abc123",
    roles: [ { role: "readWrite", db: "master" } ]
  }
)
```
For sacred, it is possible to use the admin User, but it is more secure to use a specific, more limited user.
```
MONGO_USER = "myUser"
MONGO_PW = "abc123"
DB_NAME = "master"
```

## Starting Omniboard
### without password
`omniboard -m host:27017:master`
### with password
For omniboard, I could not get it to run using the admin user (it would always connect to the admin database).
`omniboard --mu "mongodb://myUser:abc123@localhost/master?authMechanism=SCRAM-SHA-1" master`

## Connecting from cluster
I recommend using ngrok.
- `ngrok tcp 27017` to punsh a tunnel to your mongo database. Ngrok will give you an URL you can connect to.
- Use this URL as your host.




# Train
```
python3 main.py with configs.json
```

For market, you can find them [here](https://github.com/VisualComputingInstitute/triplet-reid/tree/master/data):


# Evaluation

You can use embed.py to write out embeddings that are compatible with the 
evaluation script.

```
python3 main.py evaluate_from_confipython3 main.py evaluate_experiment with evaluate.json evaluation.experiment=/dir/to/experiment
```
To calculate the final scores, please use the evaluation script from 
[here](https://github.com/VisualComputingInstitute/triplet-reid#evaluating-embeddings)!

# Scores without Re-rank (and pretrained models) 
### Market-1501
#### Trinet
Settings: 
- P=18 
- K=4
- dim=128

Download Model ([GoogleDrive](https://drive.google.com/open?id=1eNJuLxRz3dJ0MkVjoLP6vshxZUn_NLn0))

|Test time augmentation| mAP | top-1 | top-5| top-10|
|---|---:|---:|---:|---:|
| None | 65.06% | 80.31% | 92.25% | 94.71% |
| With TenCrop |  69.44% | 83.40% | 93.59% | 96.17% |


#### MGN

Settings:

```
  "K": 4,
  "P": 16,
  "checkpoint_frequency": 1000,
  "commit": "c1c1a27",
  "csv_file": "data/market1501_train.csv",
  "data_dir": "datasets/Market-1501",
  "decay_start_iteration": 15000,
  "dim": 256,
  "experiment": "new_mgn_123_single",
  "image_height": 384,
  "image_width": 128,
  "log_level": 1,
  "loss": "BatchHardSingleWithSoftmax",
  "lr": 0.0003,
  "margin": "1.2",
  "mgn_branches": [
    1,
    2,
    3
  ],
  "model": "mgn",
  "no_multi_gpu": false,
  "num_classes": 751,
  "output_path": "training",
  "restore_checkpoint": null,
  "sampler": "TripletBatchSampler",
  "scale": 1.0,
  "train_iterations": 25000
```

| Test time augmentation | mAP | top-1 | top-5| top-10|
|---|---:|---:|---:|---:|
| With Horizontal Flip | 83.17% | 93.62% | 97.86% | 98.66% |


# TODO
- [x] Evaluate current MGN
- [ ] Upload MGN model
- [ ] Improve logging (Use tensorboard or similar)
- [ ] Clean up CPU GPU mess
