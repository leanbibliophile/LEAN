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
from scheduler.task_scheduler_planner_llm import TaskSchedulerPlannerLLM
from fl_alg import LEANParamLibrary

from calflops import calculate_flops

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

from qwen_chem_utils import *

from torchvision import datasets

from settings.settings_llm import ALPHA, GPU, EVAL_GPU, device, cpu_device, fl_parameters, settings, stochastic_settings, log_folder, log_path, train_log, stats_log, FULL_RANK

from peft import get_peft_model_state_dict

QWEN_FULLRANK = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

DTYPE=torch.float32

model_name = "qwen"
version = "3-0.6B"
#version = "3-1.7B"

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

DATASET=fl_parameters["dataset"]
print("Dataset: ", DATASET)
dataset_path = {
    "chembench": "/ws/fs_mount/datasets/chembench/uspto50.pickle"
}

dataset_cache = {
    "chembench": "/ws/fs_mount/datasets/chembench/dataset_cache_reaction_prediction_uspto50_noicl.pickle"
}

def load_datasets(collator, batch_size=1, eval_size = 128, use_cache=True):
    train_data, valid_data, test_data, test_gt, valid_gt = fetch_dataset(use_cache)

    valid_loader = DataLoader(
        valid_data, batch_size=eval_size, collate_fn=collator
    )
    test_loader = DataLoader(
        test_data, batch_size=eval_size, collate_fn=collator
    )

    return train_data, valid_data, test_data, valid_loader, test_loader, valid_gt, test_gt


USE_CACHE = True

if __name__ == "__main__":
    print(fl_parameters)
    print(settings)

    # Cleaned version of FL codebase
    # Will bring this to FedVert repo too

    # Model
    print("Fetch model")
    model, tokenizer, collator = model_selector(model_name, version)

    # Dataset 
    print("Fetch data")
    train_data, valid_data, test_data, valid_loader, test_loader, valid_gt, test_gt = load_datasets(collator, use_cache=USE_CACHE)

    # FL Simulation

    # Base model
    
    local_lr = settings['lr']

    # Client models

    # Prepare data
    print("Generating samples")
    fl_parameters["use_orig_dataset"] = True # Override to always evenly partition in case of language
    if fl_parameters["use_orig_dataset"]:
        train_user_dict = partition_dataset(train_data, fl_parameters["users"])
        train_client_data = [ClientDatasetLLM(train_data, train_user_dict[u]) for u in range(fl_parameters["dataset_shards"])]
    print("Done generating samples")

    server_new_dict = {} # Empty array initially
    server_output_prev = None
    
    local_lora_rank = settings["lora_rank"]
    #central_lora_rank = min(participating_users * local_lora_rank, 1024)
    central_lora_rank = settings["central_lora_limit"]
    e_rank_list = settings["sample_lora_rank"]

    # First argument client_cnt has been decoupled and does nothing now
    print("Init library")
    fl_instance = LEANParamLibrary(fl_parameters["users_pool"], lora_rank=local_lora_rank, local_lora_rank=local_lora_rank, min_pool=settings['min_pool'], alpha=ALPHA, llm=True)

    lora_model = get_lora_model(model, local_lora_rank)
    lora.mark_only_lora_as_trainable(model)
    #model.load_state_dict(pretrained_state_dict, strict=False)
    central_model_dict = get_peft_model_state_dict(lora_model)
    #lora_prefixes = {}
    #for k in central_model_dict.keys():
    #    if "lora_A" in k:
    #        index = ".".join(k.split(".")[:-1])
    #        if index not in lora_prefixes:
    #            lora_prefixes[index] = True
    fl_instance.init_paramlib(central_model_dict)
    fl_instance.param_library.activate_init(FULL_RANK)

    plan = load_plan(settings["plan_file"])
    task_scheduler = TaskSchedulerPlannerLLM(fl_parameters["users_pool"], fl_instance, plan, collator=collator)

    #idxs_users = client_scheduler.get_uids()
    #fl_instance.update_clients(idxs_users)
    #fl_instance.client_cnt = client_scheduler.num_part

    flop_model = copy.deepcopy(lora_model)
    local_model = copy.deepcopy(lora_model)
    lora.mark_only_lora_as_trainable(local_model)
    #local_model.load_state_dict(pretrained_state_dict, strict=False)
    local_model.train()

    # Compute flop usage
    print("Calc flops")
    train_input_shape = (settings["batch_size"], QWEN_MAX_LENGTH) #get_input_shape(train_data, settings["batch_size"])
    base_batch_flops, _, _ = calculate_flops(flop_model, train_input_shape, transformer_tokenizer=tokenizer, include_backPropagation=True, output_as_string=False, output_precision=4)

    #fl_instance.partition(model=model, local_model=local_model, user_idxs=idxs_users)

    #if server_output_prev is not None:
    #    fl_instance.load(server_output_prev)

    #task_scheduler = TaskScheduler(client_scheduler.num_part, fl_instance)

    train_datasets = train_client_data

    idxs_users = list(range(fl_parameters["users_pool"]))

    eval_args = {
        #"model_class": model_class,
        "model_name": model,
        "tokenizer": tokenizer,
        "valid_loader": test_loader,
        "gt": test_gt,
        "rank_list": e_rank_list,
    }

    if fl_parameters['use_dur']:
        loc_rounds = None
    else:
        loc_rounds = fl_parameters['local_epochs']

    pretrained_state_dict = {}

    print("Execute plan")
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

    eval_logs = evaluate_llm(model, test_gt, tokenizer, device, test_loader, fl_instance, e_rank_list) 
    #eval_logs = evauate(model_class, model_name, num_classes, device, valid_loader, pretrained_state_dict, fl_instance, e_rank_list, True)
    
    for entry in eval_logs:
        e = entry["rank"]
        acc = entry["accuracy"]

        log_str = f"Eval accuracy - rank {e}: {acc}"
        print(log_str)
        with open(log_path, "a") as log_file:
            log_file.write(log_str + "\n")
        wandb.log({f"Test/acc": acc, "Eval rank": e})

