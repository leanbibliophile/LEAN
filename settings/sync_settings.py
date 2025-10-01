import os
import torch

from data_utils.utils import Logger

import wandb

GPU = [0,1,2,3]
avail_gpu = set(GPU)
EVAL_GPU = GPU[0]
device = torch.device(f'cuda:{EVAL_GPU}' if torch.cuda.is_available() else 'cpu') # Device for eval
cpu_device = torch.device('cpu')

FULL_RANK = 1280

DATASET = "birds"
FL_POP = 1 # Need to set this somehow automatically to ensure pop in plan matches
DIRIC = 1
RANK = 128
LOC_EP = 200
LORA_ALPHA = 1
PLAN_TYPE = "sync" # Always use sync
SAMPLE_SIZE = 200
LR = 6e-3
USE_PROX = False # True for fedprox
PROX = 0.003
ALG = "avg" # "avg" # Fedprox is alg+USE_PROX=True
ALG_STR = ALG
if USE_PROX:
    ALG_STR = ALG_STR + "prox"
ALG_STR = ALG_STR + f"_{PLAN_TYPE}"

freq = 25 if PLAN_TYPE == "pareto" else FL_POP
lr = 25
strag = 0.2
strag_lr = 200

ALPHA_scale = 80
ALPHA = 1/(ALPHA_scale*freq)

EXP_SETTINGS = {
    "population": FL_POP,
    "diric": DIRIC,
    "train_rank": RANK,
    "loc_ep": LOC_EP,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LR,
    "plan_file": f"/ws/fs_mount/lora_lib_plans/{PLAN_TYPE}_plan_{FL_POP}_{freq}_{lr}.txt",
    "use_prox": USE_PROX,
    "prox": PROX,
    "algorithm": ALG_STR,
    "dataset": DATASET
}

local_log_path = "/ws/fs_mount/lora_lib_logs"

print("Experiment settings:", EXP_SETTINGS)

# FL Parameters
fl_parameters = {
    "dataset": DATASET,
    "alg": ALG,
    "users": FL_POP, # Number of clients
    "users_pool": FL_POP,
    "pool_frac": 0.0625,
    "part": FL_POP, # Participation count
    "diric": DIRIC, # Dirchlet parameters
    "local_epochs": LOC_EP, # Local epochs
    "lora_alpha": LORA_ALPHA,
    "use_prox": USE_PROX,
    "prox": PROX,
    "client_samples": SAMPLE_SIZE, # Number of samples at sites
    "dataset_shards": FL_POP, # Shard dataset for IID case. Should match users_pool to ensure all clients form full dataset
    "use_orig_dataset": False, # Use original data instead of diric sampling
    "diric_replace": False, # Whether to replace elements during dirichlet sampling
    "use_orig_distr": False # Use original weighting for diric distribution
}

# Training parameters 
settings = {
    "batch_size": 8,
    "epochs": 1000,
    "lr": LR,
    "gamma": 0.7,
    "seed": 926,
    "lora_rank": RANK, #16, #128,
    "min_pool": 32, # 16, 64
    "sample_lora_rank": [RANK],
    "central_lora_limit": RANK,
    "lr_step": 50,
    "enable_stats": False,
    "plan_file": EXP_SETTINGS["plan_file"]
}

stochastic_settings = {
    "population": fl_parameters["dataset_shards"], # Match population with dataset shards
    "arr_rate": 8, # Average number of clients arriving to system per round
    "dep_rate": 1, # Number of epochs of participation
    "init": 8
}

#log_folder = "log"
#log_folder = "log_server"
log_folder = "logs_planner_test_baselines"
plan_f = (settings["plan_file"].split('/')[1]).split(".")[0]
log_path = f"./{log_folder}/lean_lora_planner_plan_{plan_f}_iid_{fl_parameters['use_orig_dataset']}_diric_{fl_parameters['diric']}_samp_{fl_parameters['client_samples']}_lora_{settings['lora_rank']}_minpool_{settings['min_pool']}.txt"
#log_path = f"./{log_folder}/lean_lora_u-{fl_parameters['users']}_part-{fl_parameters['frac']}_locep-{fl_parameters['local_epochs']}_bs-{settings['batch_size']}_lr-{settings['lr']}_lora-{settings['lora_rank']}_diric-{fl_parameters['diric']}_{settings['enable_stats']}_{fl_parameters['use_orig_dataset']}.log"
train_log = f"./{log_folder}/lean_lora_train_1.log"
stats_log = f"/ws/fs_mount/logs/lean_lora_stats.log"
os.makedirs(f"./{log_folder}", exist_ok=True)
os.makedirs("/ws/fs_mount/logs", exist_ok=True)

wandb.init(
    project = "LEAN_Exp",
    config = {
        "diric": fl_parameters["diric"],
        "lora_rank": settings["lora_rank"],
        "users": FL_POP,
        "noniid_sampling": not fl_parameters["use_orig_dataset"],
        "plan_file": settings['plan_file'],
        "min_pool": settings["min_pool"]
    },
    #id = log_path.split("/")[2],
    group = "default"
)

FULL_RANK = 1280
LOGGER = Logger(EXP_SETTINGS, [FULL_RANK], local_log_path)