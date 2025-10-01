import copy
import torch
import loralib as lora
from random import shuffle
import numpy as np
from partition.partition import *
from library.lora_param_library import LoraParameterLibrary
from library.lora_param_library_baseline import LoraParameterLibraryBaseline
from library.lora_param_library_baseline_async import LoraParameterLibraryBaselineAsync

import threading

class FedAvg():
    def __init__(self, n):
        # n number of clients participating
        self.client_cnt = n
        self.weights = [{} for _ in range(n)]

    def partition(self, model, local_model=None):
        # Just use original model
        model_dict = lora.lora_state_dict(model)
        for i in range(self.client_cnt):
            self.weights[i] = copy.deepcopy(model_dict)
    
    def aggregate(self):
        w_avg = copy.deepcopy(self.weights[0])
        for k in w_avg.keys():
            for i in range(1, self.client_cnt):
                w_avg[k] += self.weights[i][k]
            w_avg[k] = torch.div(w_avg[k], self.client_cnt)
        return w_avg

class LEAN():
    def __init__(self, n, part_cnt, lora_rank, local_lora_rank):
        # n number of clients participating
        self.client_cnt = n
        self.part_cnt = part_cnt
        self.lora_rank = lora_rank
        self.local_lora_rank = local_lora_rank
        self.weights = [{} for _ in range(part_cnt)]
        self.output_w = [{} for _ in range(part_cnt)]

        self.partition_index = list(range(self.lora_rank))

    def partition(self, model, local_model, user_idxs):
        self.model_dict = copy.deepcopy(lora.lora_state_dict(model))
        local_model_dict = copy.deepcopy(lora.lora_state_dict(local_model))

        self.user_idxs = user_idxs
        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        lora_prefixes = {}
        for k in self.model_dict.keys():
            if "lora_A" in k:
                index = ".".join(k.split(".")[:-1])
                if index not in lora_prefixes:
                    lora_prefixes[index] = True
        self.lora_prefixes = list(lora_prefixes.keys())

        #Partition index
        self.index_hidden_layer = {}
        shuffle(self.partition_index) 
        for l in self.lora_prefixes:
            self.index_hidden_layer[l] = []
            for i in range(self.part_cnt):
                index = torch.tensor(self.partition_index[i * self.local_lora_rank: (i + 1) * self.local_lora_rank])
                self.index_hidden_layer[l].append(index)

        # Model partition
        self.lora_a_weight_partition = {}
        self.lora_b_weight_partition = {}
        self.bn_weight_partition = {}
        self.bn_bias_partition = {}
        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            bn_weight = f"{p}.lora_BN.weight"
            bn_bias = f"{p}.lora_BN.bias"
            self.lora_a_weight_partition[p] = partition_FC_layer_by_output_dim_0(self.model_dict[lora_a], self.index_hidden_layer[p])
            self.lora_b_weight_partition[p] = partition_FC_layer_by_input_dim_1(self.model_dict[lora_b], self.index_hidden_layer[p])
            if bn_weight in self.model_dict and bn_bias in self.model_dict:
                self.bn_weight_partition[p], self.bn_bias_partition[p] = partition_BN_layer(self.model_dict[bn_weight], self.model_dict[bn_bias], self.index_hidden_layer[p])

        # Weight assignment
        for u_id in user_idxs:
            i_uid = invert_user_idxs[u_id]
            self.weights[i_uid] = copy.deepcopy(local_model_dict)
            for p in self.lora_prefixes:
                lora_a = f"{p}.lora_A"
                lora_b = f"{p}.lora_B"
                bn_weight = f"{p}.lora_BN.weight"
                bn_bias = f"{p}.lora_BN.bias"
                self.weights[i_uid][lora_a] = self.lora_a_weight_partition[p][i_uid]
                self.weights[i_uid][lora_b] = self.lora_b_weight_partition[p][i_uid]
                if bn_weight in self.model_dict:
                    self.weights[i_uid][bn_weight] = self.bn_weight_partition[p][i_uid]
                if bn_bias in self.model_dict:
                    self.weights[i_uid][bn_bias] = self.bn_bias_partition[p][i_uid]

    def aggregate(self):
        server_model_dict = copy.deepcopy(self.model_dict)

        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for p in self.lora_prefixes:
            layer_lora_a = []
            layer_lora_b = []
            layer_bn_weight = []
            layer_bn_bias = []

            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            bn_weight = f"{p}.lora_BN.weight"
            bn_bias = f"{p}.lora_BN.bias"

            for u_id in self.user_idxs:

                i_uid = invert_user_idxs[u_id]

                layer_lora_a.append(self.weights[i_uid][lora_a])
                layer_lora_b.append(self.weights[i_uid][lora_b])

                if bn_weight in self.model_dict:
                    layer_bn_weight.append(self.weights[i_uid][bn_weight])
                if bn_bias in self.model_dict:
                    layer_bn_bias.append(self.weights[i_uid][bn_bias])

            update_tensor_by_update_lists_dim_0(server_model_dict[lora_a], layer_lora_a, self.index_hidden_layer[p])
            update_tensor_by_update_lists_dim_1(server_model_dict[lora_b], layer_lora_b, self.index_hidden_layer[p])    
            if bn_weight in self.model_dict:
                update_tensor_by_update_lists_dim_0(server_model_dict[bn_weight], layer_bn_weight, self.index_hidden_layer[p])    
            if bn_bias in self.model_dict: 
                update_tensor_by_update_lists_dim_0(server_model_dict[bn_bias], layer_bn_bias, self.index_hidden_layer[p])    

        return server_model_dict
    
    def aggregate_statistics(self):
        layer_wise_statistics = {}

        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for l in self.lora_prefixes:
            mean_stack = [self.output_w[invert_user_idxs[u_id]][l]["first"] for u_id in self.user_idxs]
            mean = torch.mean(torch.stack(mean_stack), dim=0)

            second_stack = [self.output_w[invert_user_idxs[u_id]][l]["second"] for u_id in self.user_idxs] 
            cumulative_second = torch.mean(torch.stack(second_stack), dim=0)

            var = torch.sub(cumulative_second, torch.square(mean))

            layer_wise_statistics[f"{l}.lora_stat_mean"] = mean
            layer_wise_statistics[f"{l}.lora_stat_var"] = var

            #layer_wise_statistics[l] = {"mean": mean, "var": var}
        
        return layer_wise_statistics

class LEANLibrary():
    def __init__(self, n, part_cnt, lora_rank, local_lora_rank, max_lora_rank=1024):
        # n number of clients participating
        self.client_cnt = n
        self.part_cnt = part_cnt
        self.lora_rank = lora_rank # Total rank of all participating clients = part_cnt * loc_rank
        self.local_lora_rank = local_lora_rank # Rank at one client
        self.weights = [{} for _ in range(part_cnt)]
        self.output_w = [{} for _ in range(part_cnt)]

        self.partition_index = list(range(self.lora_rank))

    def partition(self, model, local_model, user_idxs):
        self.model_dict = copy.deepcopy(lora.lora_state_dict(model))
        local_model_dict = copy.deepcopy(lora.lora_state_dict(local_model))

        self.user_idxs = user_idxs
        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        lora_prefixes = {}
        for k in self.model_dict.keys():
            if "lora_A" in k:
                index = ".".join(k.split(".")[:-1])
                if index not in lora_prefixes:
                    lora_prefixes[index] = True
        self.lora_prefixes = list(lora_prefixes.keys())

        #Partition index
        self.index_hidden_layer = {}
        for l in self.lora_prefixes:
            self.index_hidden_layer[l] = []
            for i in range(self.part_cnt):
                index = torch.tensor(self.partition_index[i * self.local_lora_rank: (i + 1) * self.local_lora_rank])
                self.index_hidden_layer[l].append(index)

        # Model partition
        self.lora_a_weight_partition = {}
        self.lora_b_weight_partition = {}
        self.bn_weight_partition = {}
        self.bn_bias_partition = {}
        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            bn_weight = f"{p}.lora_BN.weight"
            bn_bias = f"{p}.lora_BN.bias"
            self.lora_a_weight_partition[p] = partition_FC_layer_by_output_dim_0(self.model_dict[lora_a], self.index_hidden_layer[p])
            self.lora_b_weight_partition[p] = partition_FC_layer_by_input_dim_1(self.model_dict[lora_b], self.index_hidden_layer[p])
            if bn_weight in self.model_dict and bn_bias in self.model_dict:
                self.bn_weight_partition[p], self.bn_bias_partition[p] = partition_BN_layer(self.model_dict[bn_weight], self.model_dict[bn_bias], self.index_hidden_layer[p])

        # Weight assignment
        for u_id in user_idxs:
            i_uid = invert_user_idxs[u_id]
            self.weights[i_uid] = copy.deepcopy(local_model_dict)
            for p in self.lora_prefixes:
                lora_a = f"{p}.lora_A"
                lora_b = f"{p}.lora_B"
                bn_weight = f"{p}.lora_BN.weight"
                bn_bias = f"{p}.lora_BN.bias"
                self.weights[i_uid][lora_a] = self.lora_a_weight_partition[p][i_uid]
                self.weights[i_uid][lora_b] = self.lora_b_weight_partition[p][i_uid]
                if bn_weight in self.model_dict:
                    self.weights[i_uid][bn_weight] = self.bn_weight_partition[p][i_uid]
                if bn_bias in self.model_dict:
                    self.weights[i_uid][bn_bias] = self.bn_bias_partition[p][i_uid]

    def load(self, weights_prev):
        n = min(len(self.weights), len(weights_prev)) # Prevent out of bounds loading
        self.weights[:n] = weights_prev[:n]

    def get_weights(self):
        return self.weights

    def shuffle(self, agg_dict):
        # Manually shuffle 
        # Should only be done after aggregation. Hypothetically does not affect result
        # due to commutativity 
        # agg_dict is output of self.aggregate()
        for p in self.lora_prefixes: 
            idx = torch.randperm(self.lora_rank) # Use one set of indices per layer               
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            # We don't consider bn parameters here for now
            # Shuffle A and B matrices the same way
            agg_dict[lora_a] = agg_dict[lora_a][idx, :]
            agg_dict[lora_b] = agg_dict[lora_b][:, idx]

        return agg_dict

    def aggregate(self, shuffle=False):
        server_model_dict = copy.deepcopy(self.model_dict)

        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for p in self.lora_prefixes:
            layer_lora_a = []
            layer_lora_b = []
            layer_bn_weight = []
            layer_bn_bias = []

            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            bn_weight = f"{p}.lora_BN.weight"
            bn_bias = f"{p}.lora_BN.bias"

            for u_id in self.user_idxs:

                i_uid = invert_user_idxs[u_id]

                layer_lora_a.append(self.weights[i_uid][lora_a])
                layer_lora_b.append(self.weights[i_uid][lora_b])

                if bn_weight in self.model_dict:
                    layer_bn_weight.append(self.weights[i_uid][bn_weight])
                if bn_bias in self.model_dict:
                    layer_bn_bias.append(self.weights[i_uid][bn_bias])

            update_tensor_by_update_lists_dim_0(server_model_dict[lora_a], layer_lora_a, self.index_hidden_layer[p])
            update_tensor_by_update_lists_dim_1(server_model_dict[lora_b], layer_lora_b, self.index_hidden_layer[p])    
            if bn_weight in self.model_dict:
                update_tensor_by_update_lists_dim_0(server_model_dict[bn_weight], layer_bn_weight, self.index_hidden_layer[p])    
            if bn_bias in self.model_dict: 
                update_tensor_by_update_lists_dim_0(server_model_dict[bn_bias], layer_bn_bias, self.index_hidden_layer[p])    

        if shuffle:
            server_model_dict = self.shuffle(server_model_dict)

        return server_model_dict
    
    def aggregate_statistics(self):
        layer_wise_statistics = {}

        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for l in self.lora_prefixes:
            mean_stack = [self.output_w[invert_user_idxs[u_id]][l]["first"] for u_id in self.user_idxs]
            mean = torch.mean(torch.stack(mean_stack), dim=0)

            second_stack = [self.output_w[invert_user_idxs[u_id]][l]["second"] for u_id in self.user_idxs] 
            cumulative_second = torch.mean(torch.stack(second_stack), dim=0)

            var = torch.sub(cumulative_second, torch.square(mean))

            layer_wise_statistics[f"{l}.lora_stat_mean"] = mean
            layer_wise_statistics[f"{l}.lora_stat_var"] = var

            #layer_wise_statistics[l] = {"mean": mean, "var": var}
        
        return layer_wise_statistics

class LEANParamLibrary():
    def __init__(self, n, lora_rank, local_lora_rank, min_pool, max_lora_rank=1024, alpha=1/100, llm=False):
        # n number of clients participating
        self.client_cnt = n
        self.lora_rank = lora_rank # Total rank of all participants = loc * part_cnt
        self.min_pool = min_pool
        #if llm:
        #    self.param_library = LoraParameterLibraryLLM(local_lora_rank, min_pool, alpha=alpha)
        #else:
        self.param_library = LoraParameterLibrary(local_lora_rank, min_pool, alpha=alpha, llm=llm)
        self.local_lora_rank = local_lora_rank # Local rank of one client
        self.weights = [{} for _ in range(n)]
        self.output_w = [{} for _ in range(n)]

        self.partition_index = list(range(self.lora_rank)) # Not used

        self.lock = threading.RLock()

        self.llm = llm

    def init_paramlib(self, model_dict):
        self.model_dict = copy.deepcopy(model_dict)
        lora_prefixes = {}

        for k in self.model_dict.keys():
            if "lora_A" in k:
                if self.llm:
                    index = k.split(".lora_A")[0] # This should work for both but just to be safe
                else:
                    index = ".".join(k.split(".")[:-1]) 
                if index not in lora_prefixes:
                    lora_prefixes[index] = True
        self.lora_prefixes = list(lora_prefixes.keys())
    
        self.param_library.initialize(model_dict, lora_prefixes)

    def update_clients(self, user_idxs):
        self.user_idxs = sorted(user_idxs)
        old_user_idxs_set = self.param_library.u_ids
        user_idxs_set = set(user_idxs)
        add_set = user_idxs_set.difference(old_user_idxs_set)
        remove_set = old_user_idxs_set.difference(user_idxs)
        self.client_cnt = len(user_idxs)
        
        for idx in remove_set:
            self.param_library.remove_client(idx)
        for idx in add_set:
            self.param_library.add_client(idx)

    def partition(self, model, local_model, user_idxs):
        self.weights = [{} for _ in range(self.client_cnt)]
        self.output_w = [{} for _ in range(self.client_cnt)]


        state_dict = local_model.state_dict()
        self.index_hidden_layer = self.param_library.get_index_hidden_layer()
        for idx, user_id in enumerate(user_idxs):
            self.weights[idx] = copy.deepcopy(state_dict)
            self.param_library.generate_model_params(user_id, self.weights[idx])

        #part_A, part_B = self.param_library.partition(self.index_hidden_layer)

        #self.model_dict = copy.deepcopy(lora.lora_state_dict(model))
        #local_model_dict = copy.deepcopy(lora.lora_state_dict(local_model))

        #invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        # Model partition
        #self.lora_a_weight_partition = part_A
        #self.lora_b_weight_partition = part_B
        #self.bn_weight_partition = {}
        
        # Weight assignment
        #for u_id in user_idxs:
        #    i_uid = invert_user_idxs[u_id]
        #    self.weights[i_uid] = copy.deepcopy(local_model_dict)
        #    for p in self.lora_prefixes:
        #        lora_a = f"{p}.lora_A"
        #        lora_b = f"{p}.lora_B"
        #        bn_weight = f"{p}.lora_BN.weight"
        #        bn_bias = f"{p}.lora_BN.bias"
        #        self.weights[i_uid][lora_a] = self.lora_a_weight_partition[p][i_uid]
        #        self.weights[i_uid][lora_b] = self.lora_b_weight_partition[p][i_uid]
        #        if bn_weight in self.model_dict:
        #            self.weights[i_uid][bn_weight] = self.bn_weight_partition[p][i_uid]
        #        if bn_bias in self.model_dict:
        #            self.weights[i_uid][bn_bias] = self.bn_bias_partition[p][i_uid]

    # Does not work???
    def load(self, weights_prev):
        n = min(len(self.weights), len(weights_prev)) # Prevent out of bounds loading
        self.weights[:n] = weights_prev[:n]

    def get_weights(self):
        return self.weights

    def shuffle_params(self):
        # Different manual shuffling method, using native param library method
        # This should be called BEFORE self.aggregate rather than after
        self.param_library.shuffle_params()
        # Reset index hidden layer to match new parameters
        #self.index_hidden_layer = self.param_library.get_index_hidden_layer()

    def shuffle(self, agg_dict):
        # DONT USE ???
        # Manually shuffle 
        # Should only be done after aggregation. Hypothetically does not affect result
        # due to commutativity 
        # agg_dict is output of self.aggregate()
        for p in self.lora_prefixes: 
            idx = torch.randperm(self.lora_rank) # Use one set of indices per layer               
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            # We don't consider bn parameters here for now
            # Shuffle A and B matrices the same way
            agg_dict[lora_a] = agg_dict[lora_a][idx, :]
            agg_dict[lora_b] = agg_dict[lora_b][:, idx]

        return agg_dict

    def aggregate(self, shuffle=False):
        #server_model_dict = copy.deepcopy(self.model_dict)

        invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for idx,k in invert_user_idxs.items():
            weight = copy.deepcopy(self.weights[k])
            self.param_library.update(idx, weight)

        return {} # Figure out what to do with this output?

    def aggregate_statistics(self):
        layer_wise_statistics = {}

        #invert_user_idxs = {k: idx for idx, k in enumerate(self.user_idxs)}

        for l in self.lora_prefixes:
            if l == 'fc':
                # For layers that we do not use BN at all on
                continue
            mean = torch.div(self.param_library.batch_stats[l]["sum"], self.param_library.batch_total)
            cumulative_second = torch.div(self.param_library.batch_stats[l]["sqsum"], self.param_library.batch_total)
            #mean_stack = [self.output_w[invert_user_idxs[u_id]][l]["first"] for u_id in self.user_idxs]
            #mean = torch.mean(torch.stack(mean_stack), dim=0)

            #second_stack = [self.output_w[invert_user_idxs[u_id]][l]["second"] for u_id in self.user_idxs] 
            #cumulative_second = torch.mean(torch.stack(second_stack), dim=0)

            var = torch.sub(cumulative_second, torch.square(mean))

            layer_wise_statistics[f"{l}.lora_stat_mean"] = mean
            layer_wise_statistics[f"{l}.lora_stat_var"] = var

            #layer_wise_statistics[l] = {"mean": mean, "var": var}
        
        return layer_wise_statistics
    
class BaselineParamLibrary():
    def __init__(self, n, lora_rank, local_lora_rank, min_pool=0, mode="avg", part_cnt=10, sync=True):
        self.client_cnt = n
        self.lora_rank = lora_rank # Not used        
        self.min_pool = min_pool # Not used
        if sync:
            self.param_library = LoraParameterLibraryBaseline(local_lora_rank, min_pool=min_pool, mode=mode, part_count=part_cnt)
        else:
            self.param_library = LoraParameterLibraryBaselineAsync(local_lora_rank, min_pool=min_pool, mode=mode, part_count=part_cnt)
        self.local_lora_rank = local_lora_rank # Local rank of one client
        self.weights = [{} for _ in range(n)]
        self.output_w = [{} for _ in range(n)]

        self.partition_index = list(range(self.lora_rank)) # Not used

        self.lock = threading.RLock()

    def init_paramlib(self, model_dict):
        self.model_dict = copy.deepcopy(model_dict)
        lora_prefixes = {}

        for k in self.model_dict.keys():
            if "lora_A" in k:
                index = ".".join(k.split(".")[:-1])
                if index not in lora_prefixes:
                    lora_prefixes[index] = True
        self.lora_prefixes = list(lora_prefixes.keys())
    
        self.param_library.initialize(model_dict, lora_prefixes)

    def aggregate_statistics(self):
        return {} # Dummy