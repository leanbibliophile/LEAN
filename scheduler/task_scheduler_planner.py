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

from data_utils.utils import evaluate, get_size_transfer_model
import threading
from settings.settings import GPU, fl_parameters, settings, train_log, device, cpu_device, avail_gpu, LOGGER

DTYPE=torch.float32

LOG_FREQ = 20
BATCH_LOG_FREQ = 25
USE_PROX = fl_parameters["use_prox"]
mu_prox = fl_parameters["prox"]
if USE_PROX:
    print("Using prox")
else:
    print("Not using prox")

wandb.define_metric("loc/step")
wandb.define_metric("loc/*", step_metric="loc/step")

with open(train_log, "w") as log_file:
    log_file.write(json.dumps(settings) + "\n")
    log_file.write(json.dumps(fl_parameters) + "\n")

class TaskSchedulerPlanner:
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
        self.client_checkin_cv = {id: threading.Condition() for id in range(self.client_cnt)}
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
        self.client_age = {}

        self.log_counter = 0

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
            self.local_models[i] = copy.deepcopy(model)

        while execution_flag:
            print(f"pc: {program_counter}")
            if program_counter < len(self.plan):
                command = self.plan[program_counter]
                op = command[0]
                cur_id = command[1]
                if loc_rounds is None:
                    loc_rounds = command[2]
                if op == "e":
                    age = command[2]
                    self.client_age[cur_id] = age

                    dataset = datasets[cur_id % len(datasets)]
 
                    with self.fl_instance.lock:
                        self.fl_instance.param_library.checkout(cur_id)
                        print(f"{cur_id} checkout") 
                    with self.gpu_available_cond:
                        self.gpu_available_cond.wait_for(self.gpu_avail)
                        assigned_gpu = self.avail_gpu.pop()
                        self.gpu_assignment[cur_id] = assigned_gpu 
                        
                        thread = threading.Thread(target=self.train, args=(self.local_models[assigned_gpu], local_lr, pretrained_state_dict, dataset, cur_id, self.fl_instance, assigned_gpu, loc_rounds, base_batch_flops, test_data_sample))
                        self.gpu_threads[assigned_gpu] = thread
                    with self.client_inactive_cv[cur_id]:
                        self.client_inactive[cur_id] = False   
                   
                    self.running_threads += 1
                    self.gpu_threads[assigned_gpu].start() 
                else:
                    with self.client_inactive_cv[cur_id]:
                        self.client_inactive_cv[cur_id].wait_for(lambda: self.client_done(cur_id))

                        self.gpu_assignment[cur_id] = -1   

                        # Do other processing for check-in?

                        self.running_threads -= 1
                        self.completed_runs += 1

                        with self.fl_instance.lock:
                            self.checkedin += 1
                            cur_checkins = self.checkedin
                            self.fl_instance.param_library.checkin(cur_id, self.checkin_store[cur_id]["params"], age=self.client_age[cur_id])
                            print(f"{cur_id} checkin")

                            flops = self.checkin_store[cur_id]["flops"]
                            bytes = self.checkin_store[cur_id]["bytes_transfer"]
                            train_acc = self.checkin_store[cur_id]["train acc"]
                            logs = self.checkin_store[cur_id]["batch_log"]

                            #valid_logs = self.checkin_store[cur_id]["valid_log"]

                            del self.checkin_store[cur_id]
                            wandb.log({"flops": flops, "bytes_transfer": bytes, "checkins": cur_checkins, "Train acc": train_acc}, step=cur_checkins)

                            e_rank_list = eval_args["rank_list"]
                            trained_count = 99999999
                            for l in self.fl_instance.param_library.lora_prefixes:
                                trained_count = min(trained_count, len([i for i in self.fl_instance.param_library.param_pool_ref[l] if i.active]))
                            e_rank_list = [r for r in e_rank_list if r <= trained_count]


                        if self.checkedin % LOG_FREQ == 0:
                            with self.gpu_available_cond:
                                self.gpu_available_cond.wait_for(self.gpu_avail)
                                acquired_gpu = self.avail_gpu.pop()
                                self.gpu_assignment[cur_id] = assigned_gpu 
                                eval_device = torch.device(f'cuda:{acquired_gpu}')
                            
                            model_class = eval_args["model_class"]
                            model_name = eval_args["model_name"]
                            valid_loader = eval_args["valid_loader"]
                            num_classes = eval_args["num_classes"]

                            eval_logs = evaluate(model_class, model_name, num_classes, eval_device, valid_loader, pretrained_state_dict, self.fl_instance, e_rank_list, settings["enable_stats"]) # Do not use stats
                    
                            for entry in eval_logs:
                                enable_stats = entry["enable_stats"]
                                e = entry["rank"]
                                acc = entry["accuracy"]
                                loss = entry["loss"]

                                log_str = f"Eval Valid Epoch accuracy (use stats: {enable_stats}) - rank {e}: {acc}, Epoch loss: {loss}"
                                print(log_str)
                                with open(train_log, "a") as log_file:
                                    log_file.write(log_str + "\n")
                                use_stats_str = "BN" if enable_stats else "NoBN"
                                
                                wandb.log({f"Valid/{use_stats_str}/rank_{e}/acc": acc, "Eval rank": e}, step=cur_checkins)

                                if e in LOGGER.eval_ranks:
                                    LOGGER.record(e, acc.item(), self.bytes_xfer_total, self.flop_total)

                            for rank in LOGGER.eval_ranks:
                                LOGGER.write(rank)

                                # TODO Disable test eval for now
#                             eval_test_logs = evaluate(model_class, model_name, num_classes, eval_device, test_data_sample, pretrained_state_dict, self.fl_instance, e_rank_list, True)
#                             for entry in eval_test_logs:
#                                 enable_stats = entry["enable_stats"]
#                                 e = entry["rank"]
#                                 acc = entry["accuracy"]
#                                 loss = entry["loss"]
# 
#                                 log_str = f"Eval Test Epoch accuracy (use stats: {enable_stats}) - rank {e}: {acc}, Epoch loss: {loss}"
#                                 print(log_str)
#                                 with open(train_log, "a") as log_file:
#                                     log_file.write(log_str + "\n")
#                                 use_stats_str = "BN" if enable_stats else "NoBN"
#                                 wandb.log({f"Test/{use_stats_str}/rank_{e}/acc": acc, "Eval rank": e}, step=cur_checkins)
# 


                            with self.gpu_available_cond:
                                self.avail_gpu.add(acquired_gpu)
                                self.gpu_assignment[cur_id] = -1
                                self.gpu_available_cond.notify()

            program_counter += 1
            if program_counter >= len(self.plan):
                execution_flag = False

    def train(self, model, local_lr, pretrained_state_dict, dataset, u_id, fl_instance, assigned_gpu, local_rounds, base_batch_flops=None, test_data_sample=None):
        with self.client_inactive_cv[u_id]:
            id = u_id
            #id = self.i_uids[u_id]

            assign_device = torch.device(f'cuda:{assigned_gpu}' if torch.cuda.is_available() else 'cpu')

            with fl_instance.lock:
                local_model = self.local_models[assigned_gpu]
                #local_model = copy.deepcopy(model)
                local_model_sd = lora.lora_state_dict(self.model)
                #local_model_sd = copy.deepcopy(self.model.state_dict()) 
                checkout_state_dict = fl_instance.param_library.construct_tensor(id, local_model_sd)

                transfer_model = {k: v for k,v in checkout_state_dict.items() if "lora_stat" not in k}
                self.bytes_xfer_total += get_size_transfer_model(transfer_model, dtype=DTYPE) # Download

                local_model.load_state_dict(checkout_state_dict, strict=False)
                local_model.load_state_dict(pretrained_state_dict, strict=False) 
                lora.mark_only_lora_as_trainable(local_model)
                local_model.train()
                local_model = local_model.to(assign_device)

            orig_model_params = copy.deepcopy(lora.lora_state_dict(local_model))

            # Client trainers
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(local_model.parameters(), lr=local_lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=settings["gamma"]) # I forgot to use scheduler

            #layers = get_lora_layers(local_model)

            cum_correct = 0.0
            round_loss = 0.0
            count = 0
            # Each local iteration only trains one batch as we did previously
            counter = 0
            refresh = 0
            # DEV
            batch_output_sum = {}
            batch_output_sqsum = {}
            divide_len = 0
            #first_moment_list = []
            #second_moment_list = []
            batch_logs = []

            for counter in range(local_rounds):
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

                output, lora_output = local_model(inputs)
                # DEV
                for b_id in lora_output:
                    for p in lora_output[b_id]:
                        lora_output[b_id][p] = lora_output[b_id][p].detach()
                divide_len += output.shape[0]
                for b_id in lora_output:
                    if b_id not in batch_output_sum:
                        batch_output_sum[b_id] = {}
                    if b_id not in batch_output_sqsum:
                        batch_output_sqsum[b_id] = {}
                    for p in lora_output[b_id]: 
                        if p not in batch_output_sum[b_id]:
                            batch_output_sum[b_id][p] = torch.sum(lora_output[b_id][p], dim=0).detach()
                        else:
                            batch_output_sum[b_id][p] = batch_output_sum[b_id][p].detach() + torch.sum(lora_output[b_id][p], dim=0).detach()
                        if p not in batch_output_sqsum[b_id]:
                            batch_output_sqsum[b_id][p] = torch.sum(torch.square(lora_output[b_id][p]), dim=0).detach()
                        else:
                            batch_output_sqsum[b_id][p] = batch_output_sqsum[b_id][p].detach() + torch.sum(torch.square(lora_output[b_id][p]), dim=0).detach()
                
                #w = lora_output[b_id][p]
                #first, second = get_statistics(w, dim=0)
                #first_moment_list.append(first)
                #second_moment_list.append(second)
                _, preds = torch.max(output, 1)

                # Prox
                if USE_PROX:
                    prox_term = 0.0
                    for name, param in local_model.named_parameters():
                        if name in orig_model_params:
                            prox_term += torch.linalg.norm(param - orig_model_params[name], ord=2)
#                    for k in orig_model_params:
#                        w_t = orig_model_params[k]
#                        w = local_model.state_dict()[k]
#                        prox_term += (w-w_t).norm(2)

                    loss = criterion(output, labels) + (mu_prox/2) * prox_term
                else:
                    loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)

                #round_loss += batch_loss
                #round_correct += batch_corrects
                #count += inputs.size(0)

                round_loss = batch_loss / inputs.size(0)
                cum_correct += batch_corrects.double()
                count += inputs.size(0)
                round_accuracy = batch_corrects.double() / inputs.size(0)
                cum_accuracy = cum_correct / count
                #round_accuracy = round_correct.double() / count
                #round_loss = round_loss / count

                if counter % 20 == 0: 
                    log_str = f"Train accuracy {counter}, client {u_id}: {cum_accuracy}, loss: {round_loss}"
                    print(log_str)
                    # TODO Thread safe logging
                    with open(train_log, "a") as train_l:
                        train_l.write(log_str + "\n")

                    batch_logs.append({
                        "id": u_id,
                        "batch": counter,
                        "accuracy": cum_accuracy,
                        "loss": round_loss
                    })

            #with torch.no_grad():
            #    epoch_val_accuracy = 0
            #    epoch_val_loss = 0
            #    for data, label in test_data_sample:
            #        data = data.to(assign_device)
            #        label = label.to(assign_device)
            #        
            #        val_output,_ = local_model(data)
            #        val_loss = criterion(val_output, label)

            #        acc = (val_output.argmax(dim=1) == label).float().mean()
            #        epoch_val_accuracy += acc / len(test_data_sample)
            #        epoch_val_loss += val_loss / len(test_data_sample)

            #        val_loss = criterion(val_output, label)

            #    valid_data = {"valid_loss": epoch_val_loss, "valid_acc": epoch_val_accuracy}

            del orig_model_params

            print(f"Train accuracy client {u_id}: {cum_accuracy}, loss: {round_loss}")

            self.round_accuracy[id] = cum_accuracy.item()

            with fl_instance.lock:
                for b_id in lora_output:
                    for p in lora_output[b_id]:
                        layer = f"transformer.blocks.{b_id}.attn.{p}"
                        # DEV
                        first = batch_output_sum[b_id][p].to(cpu_device)
                        second = batch_output_sqsum[b_id][p].to(cpu_device)
                        #first = torch.div(batch_output_sum[b_id][p], divide_len).to(cpu_device)
                        #second = torch.div(batch_output_sqsum[b_id][p], divide_len).to(cpu_device)
                        #w = torch.div(batch_output_sum[b_id][p], fl_parameters["local_epochs"])
                        #first, second = get_statistics(w, dim=0)
                        #first = first.to(cpu_device)
                        #second = second.to(cpu_device)
                        #fl_instance.output_w[id][layer] = {"first": first, "second": second, "bs": lr}
                        if fl_instance.param_library.batch_total == 0:
                            fl_instance.param_library.batch_stats[layer]["sum"] = first
                            fl_instance.param_library.batch_stats[layer]["sqsum"] = second
                        else:
                            fl_instance.param_library.batch_stats[layer]["sum"] = torch.add(fl_instance.param_library.batch_stats[layer]["sum"], first)                        
                            fl_instance.param_library.batch_stats[layer]["sqsum"] = torch.add(fl_instance.param_library.batch_stats[layer]["sqsum"], second)                        
                fl_instance.param_library.batch_total += divide_len

            local_model = local_model.to(cpu_device)
            with self.gpu_available_cond:
                self.avail_gpu.add(assigned_gpu)
                #self.gpu_assignment[id] = -1
                self.gpu_available_cond.notify_all()
 
            checkin_params = lora.lora_state_dict(local_model) #copy.deepcopy(lora.lora_state_dict(local_model))
            #local_model = local_model.to(assign_device)

            with fl_instance.lock:
                #fl_instance.param_library.checkin(id, checkin_params)
                
                self.flop_total += base_batch_flops * local_rounds

                transfer_model = {k: v for k,v in checkin_params.items() if "lora_stat" not in k}
                self.bytes_xfer_total += get_size_transfer_model(transfer_model, dtype=DTYPE) # Upload

                del transfer_model

                #self.checkedin += 1
                self.checkin_store[id] = {
                    "flops": self.flop_total,
                    "bytes_transfer": self.bytes_xfer_total,
                    "train acc": cum_accuracy.item(),
                    #"checkins": self.checkedin,
                    "params": checkin_params,
                    "batch_log": batch_logs
                    #"valid_log": valid_data
                }
                #wandb.log({"flops": self.flop_total, "checkins": self.checkedin, "Train acc": round_accuracy.item()}, step=self.checkedin)
                #"Train acc": round_accuracy, "epoch": counter, "client id": u_id}

                #print(fl_instance.param_library.param_pool_len)

            
            #with self.client_inactive_cv[id]:
            self.client_inactive[id] = True
            self.client_inactive_cv[id].notify()



