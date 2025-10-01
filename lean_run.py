import torch 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import loralib as lora

from data_utils.utils import *
from data_utils.dirichlet_sampler import dirichlet_sampler, dirichlet_sampler_noreplace
from client_sim.client_process import ClientProcess, load_plan

from models.model import LoraViT
from scheduler.task_scheduler_planner import TaskSchedulerPlanner
from fl_alg import LEANParamLibrary

import json
import numpy as np
import os
import random
import argparse
import copy
from tqdm import tqdm
import threading
import wandb
import pickle

from torchvision import datasets

from settings import ALPHA, GPU, EVAL_GPU, device, cpu_device, fl_parameters, settings, stochastic_settings, log_folder, log_path, train_log, stats_log, FULL_RANK

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

DTYPE=torch.float32

#model_name = "L_32"
model_name = "H_14"
#model_name = "L_32_imagenet1k"

with open(log_path, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

with open(train_log, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

with open(stats_log, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)
    return generator

generator = seed_everything(settings["seed"])

tv_dsets = [
    "cifar100",
    "stanfordcars"
]
DATASET=fl_parameters["dataset"]
print("Dataset: ", DATASET)
dataset_path = {
    "vggflowers": "/ws/fs_mount/datasets/vggflowers/images_new/",
    "birds": "/ws/fs_mount/datasets/CUB_200_2011/images_new/",
    "cifar100": "/ws/fs_mount/datasets/cifar100/",
    "stanfordcars": "/ws/fs_mount/datasets/stanfordcars/"
}

dset_classes = {
    "vggflowers": 102,
    "birds": 200,
    "cifar100": 100,
    "stanfordcars": 196
}

dataset_dir = dataset_path[DATASET]
SPLIT_RATIO = 0.2

def get_tv_dset(path, type, train=True):
    split_str = "train" if train else "test"
    if type == "cifar100":
        dset = datasets.CIFAR100(path, train=(split_str=="train"), download=False, transform=get_transforms(split_str))
        if train:
            targets = dset.targets
    elif type == "stanfordcars":
        dset = datasets.StanfordCars(path, split=split_str, download=False, transform=get_transforms(split_str))    
        targets_path = "/ws/fs_mount/datasets/stanfordcars/stanfordcars_targets.pkl"
        if train:
            with open(targets_path, "rb") as pickle_targets:
                targets = pickle.load(pickle_targets)

    if not train:
        targets = None

    return dset, targets

def tv_get_label_count(index_list, targets):
    counts = [0 for i in range(max(targets)+1)]
    subset_targets = [targets[i] for i in index_list]
    for v in subset_targets:
        counts[v] += 1
    return counts, subset_targets

def load_datasets(dataset_dir, dataset_type="image"):
    is_tv_dset = False
    tv_set_type = ""
    for dset in tv_dsets:
        if dset in dataset_dir:
            is_tv_dset = True
            tv_set_type = dset
            break

    if is_tv_dset:
        train_data_orig, targets = get_tv_dset(dataset_dir, tv_set_type, train=True)

        indices = np.arange(len(train_data_orig))
        train_list, valid_list = train_test_split(indices, test_size=SPLIT_RATIO, stratify=targets, random_state=settings["seed"])
        train_data = Subset(train_data_orig, train_list)
        valid_data = Subset(train_data_orig, valid_list)
        train_data.classes = train_data_orig.classes
        valid_data.classes = train_data_orig.classes

        train_label_count, train_targets = tv_get_label_count(train_list, targets)
        valid_label_count, valid_targets = tv_get_label_count(valid_list, targets)
        train_data.targets = train_targets
        valid_data.targets = valid_targets
        test_data, _ = get_tv_dset(dataset_dir, tv_set_type, train=False)
    else: 
        train_list, train_labels, labels_index, _ = read_traintest_dir(os.path.join(dataset_dir, "train"))
        test_list, test_labels, _, test_label_count = read_traintest_dir(os.path.join(dataset_dir, "test"))

        labels = list(labels_index.values())

        train_list, valid_list = train_test_split(train_list, test_size=SPLIT_RATIO, stratify=train_labels, random_state=settings["seed"])

        train_images, valid_images, test_images = None, None, None

        print("Loading datasets")
        # Hypothetically these should be roughly proportional and interchangeable
        train_label_count = get_label_count(train_list, labels_index)
        valid_label_count = get_label_count(valid_list, labels_index)

        if dataset_type == "image":
            train_data = ImageDataset(train_list, labels_index, transform=get_transforms("train"), images_list=train_images)
            valid_data = ImageDataset(valid_list, labels_index, transform=get_transforms("val"), images_list=valid_images)
            test_data = ImageDataset(test_list, labels_index, transform=get_transforms("test"), images_list=test_images)

    #train_loader = DataLoader(dataset=train_data, batch_size=settings["batch_size"], shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=settings["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=settings["batch_size"], shuffle=True)

    return train_data, valid_data, test_data, valid_loader, test_loader, train_label_count, valid_label_count

def model_selector(name="ViT"):
    if name == "ViT":
        return LoraViT

if __name__ == "__main__":
    print(fl_parameters)
    print(settings)

    # Cleaned version of FL codebase
    # Will bring this to FedVert repo too

    # Dataset 
    train_data, valid_data, test_data, valid_loader, test_loader, train_label_count, valid_label_count = load_datasets(dataset_dir, dataset_type="image")

    num_classes = dset_classes[DATASET]

    # FL Simulation

    # Base model
    model_class = model_selector("ViT")
    #model = model_class(model_name, pretrained=False, lora_rank=settings["lora_rank"], lora_alpha=1, enable_stats=settings['enable_stats'])
    #lora.mark_only_lora_as_trainable(model) 
    #pretrained_path = "/ws/fs_mount/model_chk/ViT/L_32.pth"
    if model_name == "H_14":
        pretrained_path = f"/ws/fs_mount/model_chk/ViT/{model_name}_aligned.pth"
    else:
        pretrained_path = f"/ws/fs_mount/model_chk/ViT/{model_name}.pth"
    pretrained_state_dict = torch.load(pretrained_path)
    # Remove classifcation head
    pretrained_state_dict.pop("fc.weight")
    pretrained_state_dict.pop("fc.bias")
    #model.load_state_dict(pretrained_state_dict, strict=False)
    
    local_lr = settings['lr']

    # Client models
    #local_argument_array = [model_name]
    #local_argument_dictionary = {"pretrained": False, "lora_rank": local_lora_rank, "lora_alpha": 1, "enable_stats": False}
    #local_model = model_class(*local_argument_array, **local_argument_dictionary)
    #local_model = model_class(model_name, pretrained=False, lora_rank=local_lora_rank, lora_alpha=1, enable_stats=False)
    #lora.mark_only_lora_as_trainable(local_model)

    #task_scheduler = TaskScheduler(fl_parameters["users"], fl_instance)

    # Prepare data
    print("Generating samples")
    if fl_parameters["use_orig_dataset"]:
        train_user_dict = partition_dataset(train_data, fl_parameters["users"])
        train_client_data = [ClientDataset(train_data, train_user_dict[u]) for u in range(fl_parameters["dataset_shards"])]
    else:
        if fl_parameters["use_orig_distr"]:
            prior_distribution = train_label_count
        else:
            prior_distribution = None
        if fl_parameters["diric_replace"]:
            train_user_dict = dirichlet_sampler(train_data, fl_parameters["users"], prior_distribution=prior_distribution, diric=fl_parameters["diric"], items_per_user=fl_parameters['client_samples'])
        else:
            train_user_dict = dirichlet_sampler_noreplace(train_data, num_users=fl_parameters["users"], prior_distribution=prior_distribution, diric=fl_parameters["diric"])
        train_client_data = [ClientDataset(train_data, train_user_dict[u]) for u in range(fl_parameters["users"])]
    print("Done generating samples")

    #part_cnt = int(fl_parameters["dataset_shards"] * fl_parameters["pool_frac"])

    #max_clients = settings["central_lora_limit"]/settings["lora_rank"]
    #client_scheduler = ClientProcess(stochastic_settings["population"], max_clients, stochastic_settings["arr_rate"], stochastic_settings["dep_rate"])
    #client_scheduler.init(stochastic_settings["init"]) # Initial participants

    server_new_dict = {} # Empty array initially
    server_output_prev = None
    
    #participating_users = client_scheduler.get_client_cnt()
    #participating_users = part_cnt
    #client_scheduler.update_client_cnt()

    local_lora_rank = settings["lora_rank"]
    #central_lora_rank = min(participating_users * local_lora_rank, 1024)
    central_lora_rank = settings["central_lora_limit"]
    e_rank_list = settings["sample_lora_rank"]

    # First argument client_cnt has been decoupled and does nothing now
    fl_instance = LEANParamLibrary(fl_parameters["users_pool"], lora_rank=local_lora_rank, local_lora_rank=local_lora_rank, min_pool=settings['min_pool'], alpha=ALPHA)

    model = model_class(model_name, num_classes=num_classes, pretrained=False, lora_rank=local_lora_rank, lean=True, lora_alpha=fl_parameters["lora_alpha"], enable_stats=settings['enable_stats'])
    lora.mark_only_lora_as_trainable(model)
    model.load_state_dict(pretrained_state_dict, strict=False)
    central_model_dict = model.state_dict()
    #lora_prefixes = {}
    #for k in central_model_dict.keys():
    #    if "lora_A" in k:
    #        index = ".".join(k.split(".")[:-1])
    #        if index not in lora_prefixes:
    #            lora_prefixes[index] = True
    fl_instance.init_paramlib(central_model_dict)
    fl_instance.param_library.activate_init(FULL_RANK)

    plan = load_plan(settings["plan_file"])
    task_scheduler = TaskSchedulerPlanner(fl_parameters["users_pool"], fl_instance, plan)

    #idxs_users = client_scheduler.get_uids()
    #fl_instance.update_clients(idxs_users)
    #fl_instance.client_cnt = client_scheduler.num_part

    flop_model = model_class(model_name, num_classes=num_classes, pretrained=False, lora_rank=local_lora_rank, lean=True, lora_alpha=1, enable_stats=False)
    local_model = model_class(model_name, num_classes=num_classes, pretrained=False, lora_rank=local_lora_rank, lean=True, lora_alpha=1, enable_stats=False)
    lora.mark_only_lora_as_trainable(local_model)
    local_model.load_state_dict(pretrained_state_dict, strict=False)
    local_model.train()

    # Compute flop usage
    train_input_shape = get_input_shape(train_data, settings["batch_size"])
    base_batch_flops, _, _ = calculate_flops(flop_model, train_input_shape, include_backPropagation=True, output_as_string=False, output_precision=4)

    #fl_instance.partition(model=model, local_model=local_model, user_idxs=idxs_users)

    #if server_output_prev is not None:
    #    fl_instance.load(server_output_prev)

    #task_scheduler = TaskScheduler(client_scheduler.num_part, fl_instance)

    train_datasets = train_client_data

    idxs_users = list(range(fl_parameters["users_pool"]))

    eval_args = {
        "model_class": model_class,
        "model_name": model_name,
        "valid_loader": valid_loader,
        "rank_list": e_rank_list,
        "num_classes": num_classes
    }

    if fl_parameters['use_dur']:
        loc_rounds = None
    else:
        loc_rounds = fl_parameters['local_epochs']

    task_scheduler.execute_plan(idxs_users, local_model, local_lr, pretrained_state_dict, train_datasets, eval_args, loc_rounds=loc_rounds, base_batch_flops=base_batch_flops, test_data_sample=test_loader)

    # LR scheduling doesnt make sense here
    # if ep + 1 % settings['lr_step']:
    #    local_lr *= settings['gamma']

    #fl_instance.shuffle_params()
    #server_new_dict = fl_instance.aggregate() # This only updates model parameters, does not generate anything to be used
    #server_shuf_dict = fl_instance.shuffle(server_new_dict)
    layer_statistics = fl_instance.aggregate_statistics() # I shouldn't need to shuffle layer statistics?
    #central_model_dict = server_new_dict | layer_statistics

    default_layer = list(fl_instance.param_library.param_pool_ref.keys())[0]
    trained_count = len([i for i in fl_instance.param_library.param_pool_ref[default_layer] if i.trained])
    print(f"library size: {trained_count}")
    e_rank_list = [r for r in e_rank_list if r <= trained_count]

    eval_logs = evaluate(model_class, model_name, num_classes, device, valid_loader, pretrained_state_dict, fl_instance, e_rank_list, True)
    
    for entry in eval_logs:
        enable_stats = entry["enable_stats"]
        e = entry["rank"]
        acc = entry["accuracy"]
        loss = entry["loss"]

        log_str = f"Eval Epoch accuracy (use stats: {enable_stats}) - rank {e}: {acc}, Epoch loss: {loss}"
        print(log_str)
        with open(log_path, "a") as log_file:
            log_file.write(log_str + "\n")
        use_stats_str = "BN" if enable_stats else "NoBN"
        wandb.log({f"Test/{use_stats_str}/acc": acc, "Eval rank": e})

