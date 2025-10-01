import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

torch.autograd.set_detect_anomaly(True) # DEBUG

from data_utils.utils import adhoc_update

import loralib as lora

import threading
import copy
import json
import wandb
import calflops

from data_utils.utils import evaluate_llm, get_size_transfer_model

import threading
from settings.async_settings_llm import GPU, fl_parameters, settings, train_log, device, cpu_device, avail_gpu, LOGGER, FULL_RANK

from peft import get_peft_model_state_dict, set_peft_model_state_dict

# TODO Update this code with modernized changes 

DTYPE=torch.float32

LOG_FREQ=5
USE_PROX=fl_parameters["use_prox"]
mu_prox = fl_parameters["prox"]
if USE_PROX:
    print("Using prox!")
else:
    print("Not using prox")

with open(train_log, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

class TaskSchedulerPlannerBaselineAsyncLLM:
    # TODO Use pytorch .distributed package
    def __init__(self, client_cnt, fl_instance, plan, collator=None):
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
        self.checkin_store = {}

        self.collator = collator

    def gpu_avail(self):
        return len(self.avail_gpu) > 0

    def client_done(self, id):
        return self.client_inactive[id]
    
    def execute_plan(self, u_ids, model, local_lr, pretrained_state_dict, datasets, eval_args, loc_rounds=None, base_batch_flops=None, test_data_sample=None):
        self.u_ids = u_ids
        program_counter = 0
        execution_flag = True

        self.model = model
        for i in self.avail_gpu:
            self.local_models[i] = copy.deepcopy(model).train()

        while execution_flag:
            print(f"pc: {program_counter}")
            if program_counter < len(self.plan):
                command = self.plan[program_counter]
                op = command[0]
                cur_id = command[1]
                if loc_rounds is None:
                    loc_rounds = command[2]
                if op == "e":
                    age = command[2] # Not used
                    dataset = datasets[cur_id % len(datasets)]
 
                    with self.fl_instance.lock:
                        self.fl_instance.param_library.checkout(cur_id)
                        print(f"{cur_id} checkout") 
                        model_snapshot = self.fl_instance.param_library.construct_tensor(cur_id, pretrained_state_dict)
                    with self.gpu_available_cond:
                        self.gpu_available_cond.wait_for(self.gpu_avail)
                        assigned_gpu = self.avail_gpu.pop()
                        self.gpu_assignment[cur_id] = assigned_gpu 

                        #self.local_models[assigned_gpu].load_state_dict(model_snapshot, strict=False)
                        #self.local_models[assigned_gpu].load_state_dict(pretrained_state_dict, strict=False)        
                        #concatenated_state_dict = copy.deepcopy(pretrained_state_dict)

                        thread = threading.Thread(target=self.train, args=(self.local_models[assigned_gpu], local_lr, model_snapshot, dataset, cur_id, self.fl_instance, assigned_gpu, loc_rounds, base_batch_flops))
                        del model_snapshot
                        
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

                        with self.fl_instance.lock:
                            self.checkedin += 1
                            cur_checkins = self.checkedin
                            self.fl_instance.param_library.checkin(cur_id, self.checkin_store[cur_id]["params"])
                            print(f"{cur_id} checkin")

                            flops = self.checkin_store[cur_id]["flops"]
                            bytes = self.checkin_store[cur_id]["bytes_transfer"]
                            #train_acc = self.checkin_store[cur_id]["train acc"]

                            del self.checkin_store[cur_id]
                            wandb.log({"flops": flops, "bytes_transfer": bytes, "checkins": cur_checkins}, step=cur_checkins)

                            while self.fl_instance.param_library.part_count < self.fl_instance.param_library.param_pool_len:
                                self.fl_instance.param_library.pop()
                            if self.fl_instance.param_library.part_count == self.fl_instance.param_library.param_pool_len:
                                self.fl_instance.param_library.synchronize() # Make sufficient number are checkedin before averaging to avoid bias

                        if self.checkedin % LOG_FREQ == 0:
                            with self.gpu_available_cond:
                                self.gpu_available_cond.wait_for(self.gpu_avail)
                                acquired_gpu = self.avail_gpu.pop()
                                self.gpu_assignment[cur_id] = assigned_gpu 
                                eval_device = torch.device(f'cuda:{acquired_gpu}')
                            
                            model_name = eval_args["model_name"]
                            tokenizer = eval_args["tokenizer"]
                            ground_truth = eval_args["gt"]
                            valid_loader = eval_args["valid_loader"] 

                            e_rank_list = [FULL_RANK]
                            eval_logs = evaluate_llm(model_name, ground_truth, tokenizer, eval_device, valid_loader, self.fl_instance, e_rank_list, flora=True)
                    
                            for entry in eval_logs:
                                #enable_stats = entry["enable_stats"]
                                e = entry["rank"]
                                acc = entry["accuracy"]
                                #loss = entry["loss"]

                                log_str = f"Eval Epoch accuracy - rank {e}: {acc}"
                                print(log_str)
                                with open(train_log, "a") as log_file:
                                    log_file.write(log_str + "\n")
                                #use_stats_str = "BN" if enable_stats else "NoBN"
                                wandb.log({f"Valid/rank_{FULL_RANK}/acc": acc, "Eval rank": e}, step=cur_checkins)

                                if e in LOGGER.eval_ranks:
                                    LOGGER.record(e, acc, self.bytes_xfer_total, self.flop_total)

                            for rank in LOGGER.eval_ranks:
                                LOGGER.write(rank)

                            with self.gpu_available_cond:
                                self.avail_gpu.add(acquired_gpu)
                                self.gpu_assignment[cur_id] = -1
                                self.gpu_available_cond.notify()

            program_counter += 1
            if program_counter >= len(self.plan):
                execution_flag = False

    @staticmethod
    def get_download_param_subset(fl_instance, state_dict):
        mode = fl_instance.param_library.mode

        param_subset = {}
        for p in fl_instance.param_library.lora_prefixes:
            lora_w = f"{p}.weight"
            lora_a, lora_b = fl_instance.param_library.get_A_B_key(p)

            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"

            # FLoRA downloads full parameters, FedAvg downloads lora set
            if mode == "avg":
                param_subset[lora_a] = state_dict[lora_a]
                param_subset[lora_b] = state_dict[lora_b]
            elif mode == "flora":
                if lora_w in state_dict: # In case classification head is not in state dict
                    param_subset[lora_w] = state_dict[lora_w]

        return param_subset

    def train(self, model, local_lr, pretrained_state_dict, dataset, u_id, fl_instance, assigned_gpu, local_rounds, base_batch_flops=None):
        with self.client_inactive_cv[u_id]: 
            id = u_id
            #id = self.i_uids[u_id]

            assign_device = torch.device(f'cuda:{assigned_gpu}' if torch.cuda.is_available() else 'cpu')

            with fl_instance.lock:
                local_model = self.local_models[assigned_gpu]
    
                #local_model = copy.deepcopy(model)
                #local_model_sd = lora.lora_state_dict(self.model)
                #local_model_sd = copy.deepcopy(self.model.state_dict()) 
                checkout_state_dict = fl_instance.param_library.flora_llm_sd(pretrained_state_dict) # Just use instance of inserted state dict because we preloaded a snapshot of model at checkout
                #checkout_state_dict = fl_instance.param_library.construct_tensor(id, pretrained_state_dict)

                download_state_dict = self.get_download_param_subset(fl_instance, checkout_state_dict)
                self.bytes_xfer_total += get_size_transfer_model(download_state_dict)

                #local_model.merge_adapter()
                local_model.load_state_dict(checkout_state_dict, strict=False) # Should it be reversed order in all cases TODO
                #adhoc_update(local_model, checkout_state_dict)
                #local_model.unmerge_adapter()
                del checkout_state_dict
                del download_state_dict
                #local_model.load_state_dict(pretrained_state_dict, strict=False) 
                lora.mark_only_lora_as_trainable(local_model)
                
                local_model.train()
                local_model = local_model.to(assign_device)

            # Prox term support, not implemented TODO
            #orig_model_params = copy.deepcopy(lora.lora_state_dict(local_model))

            # Client trainers
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(local_model.parameters(), lr=local_lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=settings["gamma"]) # I forgot to use scheduler

            #layers = get_lora_layers(local_model)


            cum_correct = 0.0
            round_correct = 0.0
            round_loss = 0.0
            count = 0
            # Each local iteration only trains one batch as we did previously
            counter = 0
            refresh = 0

            for counter in range(local_rounds):
                if refresh == 0:
                    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, collate_fn=self.collator)
                    it = iter(loader)
                    refresh = len(loader)
                #nxt_batch = next(it)
                refresh -= 1
                #batches.append(nxt_batch)

                batch = next(it)
                batch = {k: v.to(assign_device) for k, v in batch.items()}
                output = local_model(**batch)
            
                # Prox
                #if USE_PROX:
                #    prox_term = 0.0
                #    for name, param in local_model.named_parameters():
                #        if name in orig_model_params:
                #            prox_term += torch.linalg.norm(param - orig_model_params[name], ord=2)
#                    for k in orig_model_params:
#                        w_t = orig_model_params[k]
#                        w = local_model.state_dict()[k]
#                        prox_term += (w-w_t).norm(2)

                #    loss = output.loss + (mu_prox/2) * prox_term
                #else:
                loss = output.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_loss = loss

                if counter % 5 == 0: 
                    log_str = f"Train accuracy {counter}, client {u_id}: loss: {batch_loss}"
                    print(log_str)
                    # TODO Thread safe logging
                    with open(train_log, "a") as train_l:
                        train_l.write(log_str + "\n")
                #wandb.log({"Train acc": round_accuracy, "epoch": counter, "client id": u_id})

            print(f"Train accuracy client {u_id}: loss: {batch_loss}")

            #self.round_accuracy[id] = cum_accuracy.item()

            with self.gpu_available_cond:
                self.avail_gpu.add(assigned_gpu)
                #self.gpu_assignment[id] = -1
                self.gpu_available_cond.notify_all()
            
            #checkin_params = lora.lora_state_dict(local_model) #copy.deepcopy(lora.lora_state_dict(local_model))
            with fl_instance.lock:
                local_model = local_model.to(cpu_device)
                checkin_params = local_model.state_dict() #get_peft_model_state_dict(local_model)
                #fl_instance.param_library.checkin(id, checkin_params)
                
                self.flop_total += base_batch_flops * local_rounds
                transfer_model = {k: v for k,v in checkin_params.items() if "lora_stat" not in k}
                self.bytes_xfer_total += get_size_transfer_model(transfer_model, dtype=DTYPE) # Upload

                self.checkin_store[id] = {
                    "flops": self.flop_total,
                    "bytes_transfer": self.bytes_xfer_total,
                    #"train acc": cum_accuracy.item(),
                    "loss": batch_loss, 
                    "params": checkin_params
                }

                #print(self.flop_total)
                #self.checkedin += 1
                #wandb.log({"flops": self.flop_total, "checkins": self.checkedin, "Train accuracy": round_accuracy.item()}, step=self.checkedin)

                self.client_inactive[id] = True
                self.client_inactive_cv[id].notify()



