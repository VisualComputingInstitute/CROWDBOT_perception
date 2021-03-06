import torch
import torch.nn as nn
import cv2
import sys
import pickle
import datetime
import functools
import numpy as np
import os
import glob
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import signal
import pkgutil
import importlib
import operator


def restore_most_recent(path):
    most_recent = ''
    curr = 0

    for model_path in glob.glob(path + '/' + 'model_*'):
        num = int(model_path.split('_')[-1])
        if num > curr:
            most_recent = model_path

    if most_recent != '':
        return most_recent
    else:
        experiment_folders = [folder for folder in os.listdir(path) if os.path.isdir(folder) and folder.isdigit()]
        experiment_folders.reverse()
        for experiment in experiment_folders:
            for model_path in glob.glob(os.path.join(path, experiment, 'model_*')):
                num = int(model_path.split('_')[-1])
                if num > curr:
                    most_recent = model_path

            # return newest checkpoint
            if most_recent != '':
                return most_recent
        return ""

def restore_checkpoint(path, model_cfg=dict(), optimizer_cfg=dict(), map_location=None):
    """
    Args:
        path: path to the checkpoint dir or file
        model_cfg: An existing model_cfg, this will overwrite
                   the loaded cfg. If None is passed, the cfg will just
                   be restored.
    Returns:
        model_cfg: A cfg to restore the model.
        optimizer_cfg: A cfg to restore the optimizer.
        epoch: Epoch to continue counting from 0.
    """
    if os.path.isdir(path):
        file_name = restore_most_recent(path)
        if file_name == "":
            checkpoint = dict()
        else:
            checkpoint = torch.load(file_name, map_location)
    elif os.path.isfile(path):
        checkpoint = torch.load(path, map_location)
    else:
        raise ValueError("restore_checkpoint path is not a file or directory: %s" % path)


    if 'state_dict' in checkpoint:
        print("Setting weights from {}.".format(path))
        model_cfg.update({'init_from_dict': checkpoint['state_dict']})
    if 'model_cfg' in checkpoint:
        original_model_cfg = checkpoint['model_cfg']
        # overwrite saved model_cfg with passed cfg
        original_model_cfg.update(model_cfg)
        model_cfg = original_model_cfg
    elif len(model_cfg) == 0:
        raise RuntimeError('No cfg in saved model found and no cfg is passed.')

    if 'optimizer' in checkpoint:
        original_optimizer_cfg = {'init_from_dict': checkpoint['optimizer']}
        original_optimizer_cfg.update(optimizer_cfg)
        optimizer_cfg = original_optimizer_cfg
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    else:
        # TODO as warning
        print("Warning, no epoch found in model, training will start from beginning.")
        # TODO get from filename
        epoch = 0

    return model_cfg, optimizer_cfg, epoch


def var2num(x):
    return x.data.cpu().numpy()


def debug_vis(img_path, bbox=list(), pose=list()):
    cv_img_patch_show = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if len(bbox) > 0:
        c_x, c_y, width, height = bbox
        pt1 = (int(c_x - 1.0 * width / 2), int(c_y - 1.0 * height / 2))
        pt2 = (int(c_x + 1.0 * width / 2), int(c_y + 1.0 * height / 2))
        cv2.rectangle(cv_img_patch_show, pt1, pt2, (0, 128, 255), 3)

    #TODO: add flip pairs
    for pt in pose:
        if not np.any(np.isnan(pt)):
            cv2.circle(cv_img_patch_show, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
    name = tempfile.mktemp('.png')
    cv2.imwrite(name, cv_img_patch_show)
    print(name)


def visualize(image_path, points, edges, bbox=None, linewidth=8):
    """
    Args:
        List of edges: Each edge is between two joints.
    """
    if isinstance(image_path, np.ndarray):
        canvas = image_path
    else:
        canvas = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    x = points[:, 0]
    y = points[:, 1]

    for idx, edge in enumerate(edges):
        jidx1, jidx2 = edge
        pt1 = (x[jidx1], y[jidx1])
        pt2 = (x[jidx2], y[jidx2])
        if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
            continue

        # do conversion after checking for nan. Nan is special float value.
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.line(canvas, pt1, pt2, colors[idx], linewidth)

    # draw bbounding box
    if bbox is not None:
        #c_x, c_y, width, height = bbox
        #pt1 = (int(c_x - 1.0 * width / 2), int(c_y - 1.0 * height / 2))
        #pt2 = (int(c_x + 1.0 * width / 2), int(c_y + 1.0 * height / 2))
        pt1, pt2 = bbox
        cv2.rectangle(canvas, pt1, pt2, (0, 128, 255), 3)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)

    name = tempfile.mktemp()
    plt.savefig(name)
    return name


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def clean_string(s):
    """
    https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
    """
    return ''.join(e for e in s if e.isalnum())


def cache_result_on_disk(path, relevant_args=[], forced=None, min_time=None):
    """Helps with caching and restoring the results of a function call on disk.
    Specifically, it returns a function decorator that makes a function cache its result in a file.
    It only evaluates the function once, to generate the cached file. Unless forced, even if you
    then call the function with different arguments, it will still just load it from file if
    possible. The decorator also adds a new keyword argument to the function,
    called 'forced_cache_update' that also forces a regeneration of the cached file.

    Usage:
        @cache_result_on_disk('/some/path/to/a/file')
        def some_function(some_arg):
            ....

    Args:
        path: The path where the function's result is stored.
        relevant_args: Arguments that change the output of the func.
        forced: always recreate the cached version
        min_time: recreate cached file if its modification timestamp is older than this param
           The format is like 2025-12-27T10:12:32 (%Y-%m-%dT%H:%M:%S)

    Returns:
        The decorator.
    """

    if min_time is None:
        min_time = 0
    else:
        min_time = datetime.datetime.strptime(min_time, '%Y-%m-%dT%H:%M:%S').timestamp()

    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            inner_forced = forced if forced is not None else kwargs.get('forced_cache_update')
            if 'forced_cache_update' in kwargs:
                del kwargs['forced_cache_update']

            new_path = path
            for idx in relevant_args:
                arg = clean_string(args[idx])
                new_path = "{}_{}".format(new_path, arg)

            for key, value in kwargs.items():
                new_path = "{}_{}_{}".format(new_path, key, value)
            new_path = new_path + '.dat'
            if not inner_forced and os.path.exists(new_path) and os.path.getmtime(new_path) >= min_time:
                return load_pickle(new_path)
            else:
                result = f(*args, **kwargs)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                dump_pickle(result, new_path)
                return result

        return wrapped

    return decorator


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, file_path, protocol=pickle.HIGHEST_PROTOCOL):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol)


def to_cv(image):
    open_cv_image = np.array(image)
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


class ExitHandler(object):
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_routine)
        signal.signal(signal.SIGTERM, self.exit_routine)
        self._exit_routines = []
        self.is_called = False

    def exit_routine(self, signum, frame):
        # TODO NOte does not work, race condition
        if self.is_called:
            return False
        self.is_called = True

        print("Exit routine is called!")
        for routine in self._exit_routines:
            try:
                routine()
            except Exception as e:
                print(e)

        exit(0)

    def register(self, fn, *args, **kwargs):
        new_fn = functools.partial(fn, *args, **kwargs)
        self._exit_routines.append(new_fn)


def import_submodules(package_name):
    """
    From savitar2 @ P.Voigtlaender
    """
    package = sys.modules[package_name]
    for importer, name, is_package in pkgutil.walk_packages(package.__path__):
        # not sure why this check is necessary...
        if not importer.path.startswith(package.__path__[0]):
            continue
        name_with_package = package_name + "." + name
        importlib.import_module(name_with_package)
        if is_package:
            import_submodules(name_with_package)


def create_dir_recursive(path):
    """ Creates the given path if necessary """
    if path == "":
        return

    if not os.path.exists(path):
        new_path = path.split(os.path.sep)
        new_path = os.path.sep.join(new_path[:-1])
        create_dir_recursive(new_path)
        os.mkdir(path)


def format_dict_keys(dic):
    return format_string_list(dic.keys())

def format_string_list(generator):
    return ', '.join(generator)


def prod(iterator):
    return functools.reduce(operator.mul, iterator, 1)
