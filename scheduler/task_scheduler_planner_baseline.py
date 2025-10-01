import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import loralib as lora

import threading
import copy
import json
import wandb
import calflops

from tqdm import tqdm

import threading
from settings.sync_settings import GPU, fl_parameters, settings, train_log, device, cpu_device, avail_gpu, LOGGER, FULL_RANK

from data_utils.utils import evaluate, get_size_transfer_model

LOG_FREQ=20
USE_PROX=fl_parameters["use_prox"]
mu_prox = fl_parameters["prox"]
if USE_PROX:
    print("Using prox!")
else:
    print("Not using prox")

with open(train_log, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

class TaskSchedulerPlannerBaseline:
    # TODO Use pytorch .distributed package
    def __init__(self, client_cnt, fl_instance, plan):
        self.client_cnt = client_cnt
        self.fl_instance = fl_instance
        self.plan = plan

        self.gpu_available_cond = threading.Condition()
        #self.gpu_available_sema = threading.BoundedSemaphore(4)

        self.avail_gpu = avail_gpu
        avail_gpu_count = len(self.avail_gpu)
        if self.client_cnt < avail_gpu_count:
            avail_gpu_count = self.client_cnt
        avail_gpu_list = list(self.avail_gpu)[:avail_gpu_count]
        self.avail_gpu = set(avail_gpu_list)

        self.local_models = {i: None for i in self.avail_gpu}

        self.gpu_assignment = {id: -1 for id in range(self.client_cnt)}
        self.gpu_threads = {id: None for id in self.avail_gpu}
        self.client_inactive_cv = {id: threading.Condition() for id in range(self.client_cnt)}
        self.client_inactive = {id: True for id in range(self.client_cnt)}
        #self.gpu_locks = [threading.Lock() for _ in range(4)] # 1 lock for each GPU
        #self.avail_lock = set()
        self.running_threads = 0
        self.round_accuracy = [0] * client_cnt
        self.completed_runs = 0
        self.synchronization_period = self.client_cnt * 5
        self.log_dump = []

        self.flop_total = 0
        self.bytes_xfer_total = 0
        self.checkedin = 0

    def gpu_avail(self):
        return len(self.avail_gpu) > 0

    def client_done(self, id):
        return self.client_inactive[id]

    def execute_plan(self, u_ids, model, local_lr, pretrained_state_dict, datasets, eval_args, base_batch_flops=None):
        self.u_ids = u_ids
        program_counter = 0
        execution_flag = True

        self.model = model
        for i in self.avail_gpu:
            self.local_models[i] = copy.deepcopy(model)

        while execution_flag:
            print(f"pc: {program_counter}")
            if program_counter < len(self.plan):
                command = self.plan[program_counter]
                op = command[0]
                cur_id = command[1]
                lr = fl_parameters["local_epochs"]
                if op == "e":
                    dataset = datasets[cur_id % len(datasets)]
 
                    with self.fl_instance.lock:
                        self.fl_instance.param_library.checkout(cur_id)
                        print(f"{cur_id} checkout") 
                    with self.gpu_available_cond:
                        self.gpu_available_cond.wait_for(self.gpu_avail)
                        assigned_gpu = self.avail_gpu.pop()
                        self.gpu_assignment[cur_id] = assigned_gpu 
                        
                        thread = threading.Thread(target=self.train, args=(self.local_models[assigned_gpu], local_lr, pretrained_state_dict, dataset, cur_id, self.fl_instance, assigned_gpu, lr, base_batch_flops))
                        self.gpu_threads[assigned_gpu] = thread
                    with self.client_inactive_cv[cur_id]:
                        self.client_inactive[cur_id] = False   
                   
                    self.running_threads += 1
                    self.gpu_threads[assigned_gpu].start() 
                else:
                    #client_gpu = self.gpu_assignment[cur_id]
                    with self.client_inactive_cv[cur_id]:
                        self.client_inactive_cv[cur_id].wait_for(lambda: self.client_done(cur_id))

                        #self.avail_gpu.add(client_gpu)
                        self.gpu_assignment[cur_id] = -1   

                        # Do other processing for check-in?

                        self.running_threads -= 1
                        self.completed_runs += 1

                        if self.running_threads == 0:
                            with self.fl_instance.lock:
                                self.fl_instance.param_library.synchronize()

                                if self.checkedin % LOG_FREQ == 0:
                                    with self.gpu_available_cond:
                                        self.gpu_available_cond.wait_for(self.gpu_avail)
                                        acquired_gpu = self.avail_gpu.pop()
                                        self.gpu_assignment[cur_id] = assigned_gpu 
                                        eval_device = torch.device(f'cuda:{acquired_gpu}')
                                    
                                    model_class = eval_args["model_class"]
                                    model_name = eval_args["model_name"]
                                    valid_loader = eval_args["valid_loader"]
                                    e_rank_list = eval_args["rank_list"]
                                    num_classes = eval_args["num_classes"]

                                    eval_logs = evaluate(model_class, model_name, num_classes, eval_device, valid_loader, pretrained_state_dict, self.fl_instance, e_rank_list, False)
                    
                                    for entry in eval_logs:
                                        enable_stats = entry["enable_stats"]
                                        e = entry["rank"]
                                        acc = entry["accuracy"]
                                        loss = entry["loss"]

                                        log_str = f"Eval Valid Epoch accuracy (use stats: {enable_stats}) - rank {e}: {acc}, Epoch loss: {loss}"
                                        print(log_str)
                                        with open(train_log, "a") as log_file:
                                            log_file.write(log_str + "\n")
                                        # For baseline no matter what it is evaluated at full rank
                                        wandb.log({f"Valid/NoBN/rank_{FULL_RANK}/acc": acc, "Eval rank": FULL_RANK}, step=self.checkedin)

                                        if e in LOGGER.eval_ranks: 
                                            LOGGER.record(e, acc.item(), self.bytes_xfer_total, self.flop_total)

                                    for rank in LOGGER.eval_ranks:
                                        LOGGER.write(rank)

                                    with self.gpu_available_cond:
                                        self.avail_gpu.add(acquired_gpu)
                                        self.gpu_assignment[cur_id] = -1
                                        self.gpu_available_cond.notify()
            

            program_counter += 1
            if program_counter >= len(self.plan) and self.running_threads == 0:
                execution_flag = False

    @staticmethod
    def get_download_param_subset(fl_instance, state_dict):
        mode = fl_instance.param_library.mode

        param_subset = {}
        for p in fl_instance.param_library.lora_prefixes:
            lora_w = f"{p}.weight"
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"

            # FLoRA downloads full parameters, FedAvg downloads lora set
            if mode == "avg":
                param_subset[lora_a] = state_dict[lora_a]
                param_subset[lora_b] = state_dict[lora_b]
            elif mode == "flora":
                if lora_w in state_dict: # In case classification head is not in state dict
                    param_subset[lora_w] = state_dict[lora_w]

        return param_subset

    def train(self, model, local_lr, pretrained_state_dict, dataset, u_id, fl_instance, assigned_gpu, lr, base_batch_flops=None):
        id = u_id
        #id = self.i_uids[u_id]

        assign_device = torch.device(f'cuda:{assigned_gpu}' if torch.cuda.is_available() else 'cpu')

        local_model = self.local_models[assigned_gpu]
        local_model.train()
        with fl_instance.lock:
            #local_model = copy.deepcopy(model)
            #local_model_sd = lora.lora_state_dict(self.model)
            #local_model_sd = copy.deepcopy(self.model.state_dict()) 
            checkout_state_dict = fl_instance.param_library.construct_tensor(id, pretrained_state_dict)

            download_state_dict = self.get_download_param_subset(fl_instance, checkout_state_dict)
            self.bytes_xfer_total += get_size_transfer_model(download_state_dict)

            local_model.load_state_dict(checkout_state_dict, strict=False) # Should it be reversed order in all cases TODO
            del checkout_state_dict
            del download_state_dict
            #local_model.load_state_dict(pretrained_state_dict, strict=False) 
            lora.mark_only_lora_as_trainable(local_model)
            local_model = local_model.to(assign_device)

        orig_model_params = copy.deepcopy(lora.lora_state_dict(local_model))

        # Client trainers
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=local_lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=settings["gamma"]) # I forgot to use scheduler

        #layers = get_lora_layers(local_model)

        round_correct = 0.0
        round_loss = 0.0
        count = 0
        # Each local iteration only trains one batch as we did previously
        counter = 0
        refresh = 0
        for counter in range(lr):
            if refresh == 0:
                loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
                it = iter(loader)
                refresh = len(loader)
            #nxt_batch = next(it)
            refresh -= 1
            #batches.append(nxt_batch)

            inputs, labels = next(it)
            inputs = inputs.to(assign_device)
            labels = labels.to(assign_device) 

            # TODO Add regularization from HetLoRA
            output, _ = local_model(inputs)
            _, preds = torch.max(output, 1)
            
            if not USE_PROX:
                loss = criterion(output, labels)
            else:
                # Prox
                prox_term = 0.0
                for name, param in local_model.named_parameters():
                    if name in orig_model_params:
                        prox_term += torch.linalg.norm(param - orig_model_params[name], ord=2)

                loss = criterion(output, labels) + (mu_prox/2) * prox_term


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data)

            round_loss += batch_loss
            round_correct += batch_corrects
            count += inputs.size(0)

            round_accuracy = round_correct.double() / count
            round_loss = round_loss / count

            if counter % 25 == 0: 
                log_str = f"Train accuracy {counter}, client {u_id}: {round_accuracy}, loss: {round_loss}"
                print(log_str)
                # TODO Thread safe logging
                with open(train_log, "a") as train_l:
                    train_l.write(log_str + "\n")
            #wandb.log({"Train acc": round_accuracy, "epoch": counter, "client id": u_id})

        print(f"Train accuracy client {u_id}: {round_accuracy}, loss: {round_loss}")

        self.round_accuracy[id] = round_accuracy.item()

        local_model = local_model.to(cpu_device)
        with fl_instance.lock:
            checkin_params = lora.lora_state_dict(local_model) #copy.deepcopy(lora.lora_state_dict(local_model))
            fl_instance.param_library.checkin(id, checkin_params)
            
            self.flop_total += base_batch_flops * lr
            self.bytes_xfer_total += get_size_transfer_model(checkin_params) # Upload
            self.checkedin += 1
            wandb.log({"flops": self.flop_total, "bytes_transfer": self.bytes_xfer_total, "checkins": self.checkedin, "Train accuracy": round_accuracy.item()}, step=self.checkedin)

        local_model = local_model.to(assign_device)

        with self.gpu_available_cond:
            self.avail_gpu.add(assigned_gpu)
            #self.gpu_assignment[id] = -1
            self.gpu_available_cond.notify_all()
        with self.client_inactive_cv[id]:
            self.client_inactive[id] = True
            self.client_inactive_cv[id].notify()

            print(f"{id} checkin")

# Don't use
def reset_lora_params(model, prefix_list):
    for n, p in model.named_modules():
        if n in prefix_list:
            p.reset_parameters()