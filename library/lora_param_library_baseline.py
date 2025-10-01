import numpy as np
from random import shuffle
import torch.nn as nn
from partition.partition import *
import copy
from models.lean_custom_layers  import BaselineLinear
import random

from tqdm import tqdm

# For synchronous ONLY!

class ParamPair():
    def __init__(self, A, B):
        self.A = A.clone().detach()
        self.B = B.clone().detach()
        self.trained = False
    
    def set_trained(self):
        self.trained = True

    def set_A(self, v):
        self.A = v.clone().detach()

    def set_B(self, v):
        self.B = v.clone().detach()

class LoraParameterLibraryBaseline():
    def __init__(self, rank, part_count = 10, mode="avg", min_pool=0):
        self.part_count = part_count
        self.part = 0
        self.rank = rank
        self.id_to_param = {} 
        self.u_ids = set()
        self.part_uids = set()
        self.uid_to_pid = {}
        self.param_pool_ref = {} 
        self.param_template = {}
        self.rank_template = {}
        self.param_pool = {} # set(range(self.maxrank)) # Not used
        self.counter = 0
        self.param_pool_len = 0
        self.local_state_dict = {}

        self.layer_dims = {}

        self.ist_rank = self.part_count * self.rank # Only used in some algorithms

        # Mode can be "avg" or "stack" or "flora"
        self.mode = mode

        # Only used in "flora" mode:
        self.cum_w = {} 

        #self.param_pool_index = {}
        #self.param_avail_index = {} # Also marks if trained
        #self.batch_stats = {} #{"sum": None, "sqsum": None} # not used
        #self.batch_total = 0

    @staticmethod
    def extract_template(tensorA, tensorB):
        r, in_features = tensorA.shape
        out_features, r = tensorB.shape
        return in_features, out_features

    def _initialize(self, model_dict, layer_list):
        self.lora_prefixes = layer_list
        self.local_state_dict = copy.deepcopy(model_dict) 

        for l in layer_list:
            print("Initializing layer ", l)

            lora_a = f"{l}.lora_A"
            lora_b = f"{l}.lora_B" 
            self.id_to_param[l] = {}
            self.param_pool_ref[l] = []
            self.param_pool[l] = set()
            #self.param_pool[lora_a] = []
            #self.param_pool[lora_b] = []
            #self.param_pool_index[l] = []
            #self.param_avail_index[l] = set()
            in_feat, out_feat = LoraParameterLibraryBaseline.extract_template(model_dict[lora_a], model_dict[lora_b])
            self.param_template[l] = BaselineLinear(in_feat, out_feat, r=self.part_count * self.rank) # Use rank 1 LoRA to initialize parameters

            self.layer_dims[l] = [in_feat, out_feat]

            # TODO Experimental, try to fix pool size 
            # Sync
            for i in range(self.part_count * self.rank):
                #print("pool", i)
                #self.param_template[l].reset_parameters()
                new_params = self.param_template[l].state_dict()
                #l_a = copy.deepcopy(new_params['lora_A'])
                #l_b = copy.deepcopy(new_params['lora_B']) 

                l_a = torch.reshape(new_params["lora_A"][i], (1, in_feat))
                l_b = torch.reshape(new_params["lora_B"][:,i], (out_feat, 1))

                self.param_pool_ref[l].append(ParamPair(l_a, l_b))
        
        self.param_pool_len = self.part_count * self.rank

    def initialize(self, model_dict, layer_list): # Old initialization method
        self.lora_prefixes = layer_list
        for l in layer_list:
            lora_a = f"{l}.lora_A"
            lora_b = f"{l}.lora_B" 
            self.id_to_param[l] = {}
            self.param_pool_ref[l] = []
            self.param_pool[l] = []

            #self.param_pool[lora_a] = []
            #self.param_pool[lora_b] = []
            #self.param_pool_index[l] = []
            #self.param_avail_index[l] = set()
            #self.batch_stats[l] = {"sum": None, "sqsum": None}
            in_feat, out_feat = LoraParameterLibraryBaseline.extract_template(model_dict[lora_a], model_dict[lora_b])
            self.layer_dims[l] = [in_feat, out_feat]
            self.param_template[l] = BaselineLinear(in_feat, out_feat, r=self.rank) # Template for model
            self.rank_template[l] = BaselineLinear(in_feat, out_feat, r=self.rank)

            #if self.mode == "flora":
            #    weight_name = f"{l}.weight"
            #    self.cum_w[weight_name] = copy.deepcopy(self.param_template[l].weight)

        self.local_state_dict = copy.deepcopy(model_dict) #{copy.deepcopy(model_dict[l]) for l in layer_list}

        #if self.mode == "flora":

        for i in tqdm(range(self.part_count)):
            indices = self.create_new_params()
            # Do I need to do anything with this index?

    def load_initial_frozen_weights(self, state_dict):
        if self.mode != "flora":
            return
        for p in self.lora_prefixes:
            if p == "fc":
                continue # Skip classification head
            weight = f"{p}.weight"
            self.local_state_dict[weight] = state_dict[weight]

    def create_new_params(self):
        indices = {}
        for p in self.lora_prefixes:
            if p not in indices:
                indices[p] = []
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            self.param_template[p].reset_parameters()
            new_params = self.param_template[p].state_dict()

            dims = self.layer_dims[p]

            for i in range(self.rank):
                #print("pool", i)
                #self.param_template[l].reset_parameters()
                #l_a = copy.deepcopy(new_params['lora_A'])
                #l_b = copy.deepcopy(new_params['lora_B']) 
                index = self.counter + i 
                indices[p].append(index)

                l_a = torch.reshape(new_params["lora_A"][i], (1, dims[0]))
                l_b = torch.reshape(new_params["lora_B"][:,i], (dims[1], 1))

                self.param_pool_ref[p].append(ParamPair(l_a, l_b))
                self.param_pool[p].append(index)

        self.counter += self.rank
        self.param_pool_len += self.rank

    def _create_new_params(self):
        indices = {}
        for i in range(self.rank):
            for p in self.lora_prefixes:
                if p not in indices:
                    indices[p] = []
                lora_a = f"{p}.lora_A"
                lora_b = f"{p}.lora_B"
                self.param_template[p].reset_parameters()
                new_params = self.param_template[p].state_dict()
                #l_a = copy.deepcopy(new_params["lora_A"])
                #l_b = copy.deepcopy(new_params["lora_B"])
                index = self.counter
                indices[p].append(index)
                self.param_pool_ref[p].append(ParamPair(new_params['lora_A'], new_params['lora_B']))
                #self.param_pool_ref[p].append(ParamPair(l_a, l_b))
                self.param_pool[p].append(index)
            self.counter += 1
            self.param_pool_len += 1
        return indices

    def construct_tensor(self, id, state_dict):
        if self.mode == "avg" or self.mode == "flora":
            output_dict = state_dict
            for p in self.lora_prefixes:
                lora_a = f"{p}.lora_A"
                lora_b = f"{p}.lora_B"

                output_dict[lora_a] = self.local_state_dict[lora_a]
                output_dict[lora_b] = self.local_state_dict[lora_b]

                if self.mode == "flora":
                    lora_w = f"{p}.weight"
                    if lora_w in self.local_state_dict: # To account for uninitialized FC head
                        output_dict[lora_w] = self.local_state_dict[lora_w] # self.cum_w[lora_w]
                    self.rank_template[p].reset_parameters()
                    output_dict[lora_a] = self.rank_template[p].lora_A
                    output_dict[lora_b] = self.rank_template[p].lora_B

            return output_dict

        subset_param_dict = state_dict # copy.deepcopy(state_dict)
        subsample_index_hidden_layer = {}
        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"
            param_idxs = self.id_to_param[p][id]
            subsample_index_hidden_layer[p] = [param_idxs]

            construct_index_hl = torch.tensor([i for i in range(len(param_idxs))])

            A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
            B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
            update_tensor_by_update_lists_dim_0(subset_param_dict[lora_a], A_list, construct_index_hl)
            update_tensor_by_update_lists_dim_1(subset_param_dict[lora_b], B_list, construct_index_hl)
        return subset_param_dict

    def checkout(self, id):
        # For averaging mode, checkout step is just to determine which indices of "paramlib" to update
        # Will always load self.local_state_dict
        if self.part == self.part_count:
            # Ignore request if we hit participation limit
            return 
        #state_dict = {}
        if id not in self.part_uids:
            #index = self.create_new_params() # Generate parameters for new client
            #for p in self.lora_prefixes:
            #    self.id_to_param[p][id] = index[p]
            self.part_uids.add(id) # Register new participant
        pid = self.part
        #self.part_uids[pid] = id
        #self.uid_to_pid[id] = pid
        self.part += 1

        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"  

            pool_list = np.array(list(self.param_pool[p]))
            param_idxs = pool_list[pid * self.rank: (pid+1) * self.rank]
            #param_idxs = np.random.choice(pool_list, self.rank, replace=False)
            #self.param_pool[p].difference_update(param_idxs)
            self.id_to_param[p][id] = param_idxs.tolist()

        #self.param_pool_len -= self.rank
        #self.create_new_params()

    def checkin(self, id, state_dict):
        if id not in self.part_uids:
            # Ignore spurious requests
            return 

        for p in self.lora_prefixes:
            param_idxs = self.id_to_param[p].pop(id, None)
            #self.param_pool[p].update(param_idxs) # Reintroduce parameters to pool

            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B" 
            index_hidden_layer = torch.tensor(param_idxs)
            construct_index_hl = torch.tensor([i for i in range(self.rank)])
            part_a = partition_FC_layer_by_output_dim_0(state_dict[lora_a], construct_index_hl)
            part_b = partition_FC_layer_by_input_dim_1(state_dict[lora_b], construct_index_hl)
            for i, idx in enumerate(index_hidden_layer):
                # It's okay if we screw up the order :D
                self.param_pool_ref[p][idx].set_A(part_a[i])
                self.param_pool_ref[p][idx].set_B(part_b[i])
                self.param_pool_ref[p][idx].set_trained()

        self.param_pool_len += self.rank
        self.part_uids.remove(id)
        self.part -= 1

        #if self.part == 0:
        #    self.synchronize()

    def synchronize(self):
        if self.mode == "stack":
            # Shuffling
            for p in self.lora_prefixes:
                random.shuffle(self.param_pool[p])
                lora_A = f"{p}.lora_A"
                lora_B = f"{p}.lora_B"

                index_hidden_lyaer = torch.tensor(self.param_pool[p])
                A_list = [self.param_pool_ref[p][idx].A for idx in self.param_pool[p]]
                B_list = [self.param_pool_ref[p][idx].B for idx in self.param_pool[p]]
                update_tensor_by_update_lists_dim_0(self.local_state_dict[lora_A], A_list, index_hidden_lyaer)
                update_tensor_by_update_lists_dim_1(self.local_state_dict[lora_B], B_list, index_hidden_lyaer)
        if self.mode == "flora":
            state_dict_template = copy.deepcopy(self.local_state_dict)
            construct_index = torch.tensor([i for i in range(self.rank)])
            for p in self.lora_prefixes:
                lora_A_agg = []
                lora_B_agg = []
                lora_A = f"{p}.lora_A"
                lora_B = f"{p}.lora_B"
                param_A = state_dict_template[lora_A]
                param_B = state_dict_template[lora_B]
                for pid in range(self.part_count):
                    param_idxs = range(pid * self.rank, (pid+1) * self.rank)
                    A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
                    B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
                    update_tensor_by_update_lists_dim_0(param_A, A_list, construct_index)
                    update_tensor_by_update_lists_dim_1(param_B, B_list, construct_index)
                    lora_A_agg.append(copy.deepcopy(param_A))
                    lora_B_agg.append(copy.deepcopy(param_B))
                stack_A = torch.cat(lora_A_agg, dim=0) # Use cat instead of stack for this
                stack_B = torch.cat(lora_B_agg, dim=1)

                weight = f"{p}.weight"
                scaling = 1.0 / self.rank # Figure out how to extract this?
                avg_delta_w = (stack_B @ stack_A).div(self.part_count).mul(scaling)
                # Don't update local_state_dict?
                #self.cum_w[weight] = self.cum_w[weight].add(avg_delta_w) 

                if weight in self.local_state_dict: # In case of FC layer
                    self.local_state_dict[weight] = self.local_state_dict[weight].add(avg_delta_w) #self.cum_w[weight] # Do I update here? 
                else:
                    self.local_state_dict[weight] = avg_delta_w # Is this the correct approach for FC layer?
        else: # Avg
            # Do averaging here?
            state_dict_template = copy.deepcopy(self.local_state_dict)
            construct_index = torch.tensor([i for i in range(self.rank)])
            for p in self.lora_prefixes:
                lora_A_agg = []
                lora_B_agg = []
                lora_A = f"{p}.lora_A"
                lora_B = f"{p}.lora_B"
                param_A = state_dict_template[lora_A]
                param_B = state_dict_template[lora_B]
                for pid in range(self.part_count):
                    param_idxs = range(pid * self.rank, (pid+1) * self.rank)
                    A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
                    B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
                    update_tensor_by_update_lists_dim_0(param_A, A_list, construct_index)
                    update_tensor_by_update_lists_dim_1(param_B, B_list, construct_index)
                    lora_A_agg.append(copy.deepcopy(param_A))
                    lora_B_agg.append(copy.deepcopy(param_B))
                stack_A = torch.stack(lora_A_agg, dim=0)
                stack_B = torch.stack(lora_B_agg, dim=0)
                avg_A = torch.sum(stack_A, dim=0).div(self.part_count)
                avg_B = torch.sum(stack_B, dim=0).div(self.part_count)

                self.local_state_dict[lora_A] = copy.deepcopy(avg_A)
                self.local_state_dict[lora_B] = copy.deepcopy(avg_B)

                #part_a = partition_FC_layer_by_output_dim_0(avg_A, construct_index_hl)
                #part_b = partition_FC_layer_by_input_dim_1(avg_B, construct_index_hl)
                #for idx in range(self.rank):
                #   for pid in range(self.part_count):
                #        self.param_pool_ref[p][idx].set_A(part_a[i])
                #        self.param_pool_ref[p][idx].set_B(part_b[i])

    def add_client(self, id):
        p0 = [p for p in self.lora_prefixes][0]
        if len(self.param_avail_index[p0]) == 0:
            update_flag = True
            self.param_pool_len += self.rank
        else:
            update_flag = False

        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"  
            if update_flag:
                param_idxs = []
                # Generate new parameters
                for i in range(self.rank):
                    self.param_template[p].reset_parameters()
                    new_params = self.param_template[p].state_dict()
                    l_a = copy.deepcopy(new_params["lora_A"])
                    l_b = copy.deepcopy(new_params["lora_B"])
                    self.param_pool[lora_a].append(l_a)
                    self.param_pool[lora_b].append(l_b)
                    self.param_pool_index[p].append(0)
                    param_idxs.append(len(self.param_pool_index[p]) - 1)
                self.id_to_param[p][id] = param_idxs
            else:
                pool_list = np.array(list(self.param_avail_index[p]))
                param_idxs = np.random.choice(pool_list, self.rank, replace=False)
                self.param_avail_index[p].difference_update(param_idxs)
                self.id_to_param[p][id] = param_idxs.tolist()

            #for idx in param_idxs:
            #    self.param_owner[p][idx] = id
                #self.trained[idx] = 1

        self.u_ids.add(id)
        self.part -= 1
        #if self.part == 0:
        #    self.synchronize()

    def generate_model_params(self, id, state_dict):
        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B"    
            index_hidden_layer = torch.tensor(self.id_to_param[p][id])

            A_list = [self.param_pool[lora_a][idx] for idx in index_hidden_layer]
            B_list = [self.param_pool[lora_b][idx] for idx in index_hidden_layer]

            construct_index_hl = torch.tensor([i for i in range(len(index_hidden_layer))])

            update_tensor_by_update_lists_dim_0(state_dict[lora_a], A_list, construct_index_hl)
            update_tensor_by_update_lists_dim_1(state_dict[lora_b], B_list, construct_index_hl) 
        return state_dict 

    def update(self, id, state_dict):
        for p in self.lora_prefixes:
            lora_a = f"{p}.lora_A"
            lora_b = f"{p}.lora_B" 
            index_hidden_layer = torch.tensor(self.id_to_param[p][id])
            construct_index_hl = torch.tensor([i for i in range(len(index_hidden_layer))])
            part_a = partition_FC_layer_by_output_dim_0(state_dict[lora_a], construct_index_hl)
            part_b = partition_FC_layer_by_input_dim_1(state_dict[lora_b], construct_index_hl)
            for i, idx in enumerate(index_hidden_layer):
                # It's okay if we screw up the order :D
                self.param_pool[lora_a][idx] = part_a[i]
                self.param_pool[lora_b][idx] = part_b[i]
                self.param_pool_index[p][idx] = 1

    def remove_client(self, id):
        for p in self.lora_prefixes:
            if id not in self.id_to_param[p]:
                continue
            idx_list = self.id_to_param[p][id]
            self.id_to_param[p].pop(id)
            for idx in idx_list:
                # Release parameters
                self.param_avail_index[p].add(idx)

        self.u_ids.remove(id) 
        self.part -= 1        

    def shuffle_params(self):
        for p in self.lora_prefixes:
            user_idx = list(self.id_to_param[p].keys())
            shuffle_list = []
            for id in user_idx:
                shuffle_list.extend(self.id_to_param[p][id])
            shuffle(shuffle_list)

            for index, id in enumerate(user_idx):
                self.id_to_param[p][id] = shuffle_list[index * self.rank: (index + 1) * self.rank]
                #for i in shuffle_list[index * self.rank: (index + 1) * self.rank]:
                    #self.param_owner[p][i] = id

    def get_params(self):
        return self.id_to_param

    def get_index_hidden_layer(self):
        index_hidden_layer = {}
        for p in self.lora_prefixes:
            index_hidden_layer[p] = []
            clients_list = list(self.id_to_param[p].keys()) # List of client ids
            clients_list = sorted(clients_list)
            for c in clients_list:
                index_hidden_layer[p].append(torch.tensor(self.id_to_param[p][c]))
        return index_hidden_layer

    def get_rank_subset(self, rank, state_dict):
        out_state_dict = self.local_state_dict
        #if self.mode == "flora":
        #    for k,v in self.cum_w.items():
        #        out_state_dict[k] = v
        return out_state_dict, {}

#         # Only sample from columns that are trained
#         subset_param_dict = copy.deepcopy(state_dict)
#         subsample_index_hidden_layer = {}
#         # Debug
#         #u0 = list(self.u_ids)[0]
#         for p in self.lora_prefixes:
#             lora_a = f"{p}.lora_A"
#             lora_b = f"{p}.lora_B"
#             trained_params = [idx for idx, v in enumerate(self.param_pool_ref[p]) if v.trained]
#             #trained_params = [idx for idx in range(self.maxrank) if self.trained[p][idx]==1]
#             param_idxs = np.random.choice(np.array(trained_params), rank, replace=False)    
#             #param_idxs = self.id_to_param[p][u0]
#             subsample_index_hidden_layer[p] = [param_idxs]
# 
#             construct_index_hl = torch.tensor([i for i in range(len(param_idxs))])
# 
#             A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
#             B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
#             update_tensor_by_update_lists_dim_0(subset_param_dict[lora_a], A_list, construct_index_hl)
#             update_tensor_by_update_lists_dim_1(subset_param_dict[lora_b], B_list, construct_index_hl)
#             #subset_param_dict[lora_a] = partition_FC_layer_by_output_dim_0(self.A, subsample_index_hidden_layer[p])[0]
#             #subset_param_dict[lora_b] = partition_FC_layer_by_input_dim_1(self.B, subsample_index_hidden_layer[p])[0]
#         return subset_param_dict, subsample_index_hidden_layer
#     
#     #def get_batch_statistics(self):
# 