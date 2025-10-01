import os
import torch

from data_utils.utils import Logger

import wandb

import os


GPU = [0,1,2,3]
avail_gpu = set(GPU)
EVAL_GPU = GPU[0]
device = torch.device(f'cuda:{EVAL_GPU}' if torch.cuda.is_available() else 'cpu') # Device for eval
cpu_device = torch.device('cpu')

FULL_RANK = 1024

DATASET = "ChemBenchLLM"
FL_POP = 25 # Need to set this somehow automatically to ensure pop in plan matches
# skew = 0.01; semi = 1; iid=100 (no need to run this)
WIN_LENGTH = 5 # 20
DIRIC = 1
RANK = 64
LOC_EP = 50
LORA_ALPHA = 32
PARETO = 1.16 # 1.16, 0.7
PLAN_TYPE = "async" # "async0.9"
PLAN_TYPE = f"{PLAN_TYPE}{PARETO}"
SAMPLE_SIZE = 200
LR = 3e-4
USE_PROX = False
PROX = 0.003
ALG = "flora" # "avg" # Fedprox is alg+USE_PROX=True
ALG_STR = ALG
if USE_PROX:
    ALG_STR = ALG_STR + "prox"
ALG_STR = ALG_STR + f"_{PLAN_TYPE}"

freq = FL_POP # 25 if PLAN_TYPE == "pareto" or "async" in PLAN_TYPE else FL_POP
lr = 25
strag = 0.2
strag_lr = 200

ALPHA_scale = 80
ALPHA = 1/(ALPHA_scale*freq)

EXP_SETTINGS = {
    "alg": ALG_STR,
    "population": FL_POP,
    "diric": DIRIC,
    "train_rank": RANK,
    "loc_ep": LOC_EP,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LR,
    "plan_file": f"/ws/fs_mount/lora_lib_plans/{PLAN_TYPE}_plan_{FL_POP}_{freq}_{lr}.txt",
    "use_prox": USE_PROX,
    "prox": PROX,
    "algorithm": ALG,
    "dataset": DATASET,
    "window_length": WIN_LENGTH,
}

local_log_path = "/ws/fs_mount/lora_lib_logs"


# FL Parameters
fl_parameters = {
    "alg": ALG,
    "dataset": DATASET,
    "users": FL_POP, # Number of clients
    "users_pool": FL_POP,
    "pool_frac": 0.0625,
    "part": WIN_LENGTH,
    "frac": 1, # Participation ratio
    "diric": DIRIC, # Dirchlet parameters
    "local_epochs": LOC_EP, # Local epochs
    "lora_alpha": LORA_ALPHA,
    "use_prox": USE_PROX,
    "prox": PROX,
    "use_dur": False, # Whether to use plan duration or local epochs
    "client_samples": SAMPLE_SIZE, # Number of samples at sites
    "dataset_shards": FL_POP, # Shard dataset for IID case. Should match users_pool to ensure all clients form full dataset
    "use_orig_dataset": False, # Use original data instead of diric sampling
    "diric_replace": False, # Whether to replace elements during dirichlet sampling
    "use_orig_distr": False # Use original weighting for diric distribution
}

import math
#MINPOOL = 640
#MINPOOL = RANK * FL_POP
MINPOOL = 128 * FL_POP
i = int(math.floor(math.log(MINPOOL, 2)))
two_powers = [2**e for e in range(3, i+1)]
if two_powers[-1] != MINPOOL:
    two_powers.append(MINPOOL)

if MINPOOL == FULL_RANK:
    eval_pool = [FULL_RANK]
else:
    # eval_pool = [FULL_RANK, int(MINPOOL)]
    eval_pool = [FULL_RANK, 2 * FULL_RANK, 3 * FULL_RANK]

# Training parameters 
settings = {
    "batch_size": 4,
    "epochs": 1000,
    "lr": LR,
    "gamma": 0.7,
    "seed": 926,
    "lora_rank": RANK, #16, #128,
    "min_pool": MINPOOL, # 16, 64
    "sample_lora_rank": eval_pool, #two_powers, # [8, 16, 32, 64], # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "central_lora_limit": FULL_RANK,
    "lr_step": 50,
    "enable_stats": False, # TODO For now disable stats setting 
    "plan_file": f"/ws/fs_mount/lora_lib_plans/{PLAN_TYPE}_plan_{FL_POP}_{freq}_{lr}.txt"
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

LOGGER = Logger(EXP_SETTINGS, eval_pool, local_log_path)


print("Experiment settings:", EXP_SETTINGS)