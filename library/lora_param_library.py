import numpy as np
from random import shuffle
import torch.nn as nn
from partition.partition import *
import copy
from models.lean_custom_layers  import LEANStatLinear
from tqdm import tqdm

USE_DISTR_CHECKIN = False

class ParamPair():
    def __init__(self, A, B):
        self.A = A.clone().detach()
        self.B = B.clone().detach()
        self.active = False
        self.checkout = False

    def set_active(self):
        self.active = True

    def checkout(self):
        self.checkout = True
    
    def checkin(self):
        self.checkout = False

    def set_A(self, v):
        self.A = v.clone().detach()

    def set_B(self, v):
        self.B = v.clone().detach()

class LoraParameterLibrary():
    # Should have one Parameter Library per layer
    def __init__(self, rank, min_pool, alpha=1/25, llm=False):
        self.rank = rank
        self.min_pool = min_pool #max(min_pool, self.rank)
        self.alpha = alpha
        self.id_to_param = {} 
        self.u_ids = set()
        self.part = 0
        self.counter = 0
        self.param_pool_ref = {}
        self.param_pool = {} # set(range(self.maxrank))
        self.param_pool_len = 0
        self.param_pool_total = 0
        #self.param_pool_index = {}
        #self.param_avail_index = {} # Also marks if trained
        self.param_template = {}
        self.batch_stats = {} #{"sum": None, "sqsum": None}
        self.batch_total = 0

        self.active_count = 0
        self.param_pool_active_flag = {}
        self.param_pool_checkout_flag = {}

        self.llm = llm

    @staticmethod
    def extract_template(tensorA, tensorB):
        r, in_features = tensorA.shape
        out_features, r = tensorB.shape
        return in_features, out_features

    def get_A_B_key(self, layer):
        if self.llm:
            return f"{layer}.lora_A.weight", f"{layer}.lora_B.weight"
            #return f"{layer}.lora_A.default.weight", f"{layer}.lora_B.default.weight"
        else: 
            return f"{layer}.lora_A", f"{layer}.lora_B" 

    def initialize(self, model_dict, layer_list):
        self.lora_prefixes = layer_list
        for l in layer_list:
            print("Initializing layer ", l)

            lora_a, lora_b  = self.get_A_B_key(l)

            #lora_a = f"{l}.lora_A"
            #lora_b = f"{l}.lora_B" 
            self.id_to_param[l] = {}
            self.param_pool_ref[l] = []
            self.param_pool[l] = set()
            #self.param_pool[lora_a] = []
            #self.param_pool[lora_b] = []
            #self.param_pool_index[l] = []
            #self.param_avail_index[l] = set()
            self.batch_stats[l] = {"sum": None, "sqsum": None}
            in_feat, out_feat = LoraParameterLibrary.extract_template(model_dict[lora_a], model_dict[lora_b])
            self.param_template[l] = LEANStatLinear(in_feat, out_feat, r=self.min_pool) # Use rank 1 LoRA to initialize parameters

            # TODO Experimental, try to fix pool size 
            # Sync
            for i in range(self.min_pool):
                #print("pool", i)
                #self.param_template[l].reset_parameters()
                new_params = self.param_template[l].state_dict()
                #l_a = copy.deepcopy(new_params['lora_A'])
                #l_b = copy.deepcopy(new_params['lora_B']) 

                l_a = torch.reshape(new_params["lora_A"][i], (1, in_feat))
                l_b = torch.reshape(new_params["lora_B"][:,i], (out_feat, 1))

                self.param_pool_ref[l].append(ParamPair(l_a, l_b))
                self.param_pool_len += 1
                self.param_pool_total += 1

            self.param_pool_checkout_flag[l] = [0] * self.min_pool # Sync
            self.param_pool_active_flag[l] = [0] * self.min_pool
            self.active_count = 0
            self.checkout_count = 0

        #self.create_new_params() # Async

    def activate_init(self, init_pool):
        for l in self.lora_prefixes:
            for i in range(init_pool):
                self.param_pool_active_flag[l][i] = 1
                self.param_pool_ref[l][i].set_active()

        self.active_count = init_pool

    def get_active_set(self):
        active_set = {}
        for p in self.lora_prefixes:
            active_set[p] = set((i for i in range(self.min_pool) if self.param_pool_active_flag[p][i] == 1))
        return active_set

    def get_avail_set(self):
        avail_set = {}
        active_set = self.get_active_set()
        for p in self.lora_prefixes:
            avail_set[p] = set((i for i in range(self.min_pool) if (i in active_set[p] and self.param_pool_checkout_flag[p][i] == 0)))
        return avail_set

    def create_new_params(self):
        if self.active_count >= self.min_pool:
            print("Error: base parameter pool too small")
            return # Can't create more parameters
        add_count = self.active_count - self.checkout_count + self.rank # target_size - self.param_pool_len 

        if self.active_count + add_count > self.min_pool:
            add_count = self.min_pool - self.active_count # Maxed out

        added = 0
        while added < add_count: 
            for p in self.lora_prefixes:
                # Select paramter to copy
                idx_pick = np.random.choice(self.active_count) 
                copy_param = self.param_pool_ref[p][idx_pick]

                # Activate paramter
                self.param_pool_active_flag[p][self.active_count + added] = 1 

                # Assign
                self.param_pool_ref[p][self.active_count + added].set_A(copy_param.A)
                self.param_pool_ref[p][self.active_count + added].set_B(copy_param.B)
                self.param_pool_ref[p][self.active_count + added].set_active()

            added += 1

        self.active_count += add_count

    # Old 
    def _create_new_params(self):
        #target_size = self.min_pool + self.rank
        add_count = self.active_count - self.checkout_count + self.rank # target_size - self.param_pool_len 
        pool_addition = {} # Add them all at once to avoid biasing in favor of one particular parameter
        index_addition = {}
        for p in self.lora_prefixes:
            pool_addition[p] = []
            index_addition[p] = []
        added = 0

        while added < add_count: 
            for p in self.lora_prefixes:
                lora_a, lora_b  = self.get_A_B_key(p)

                #lora_a = f"{p}.lora_A"
                #lora_b = f"{p}.lora_B" 
                pool_avail = list(self.param_pool[p])
                pool_avail = [i for i in pool_avail if self.param_pool_ref[p][i].active]
                pool_avail = np.array(pool_avail)
                #pool_avail = [] # Disable this feature for now to test
                if len(pool_avail) > 0:
                    pool_pick = np.random.choice(pool_avail)
                    param_pick = self.param_pool_ref[p][pool_pick]
                    state_dict = {"lora_A": param_pick.A, "lora_B": param_pick.B}
                    self.param_template[p].load_state_dict(state_dict, strict=False)
                    
                    new_params = self.param_template[p].state_dict()
                    l_a = copy.deepcopy(new_params['lora_A'])
                    l_b = copy.deepcopy(new_params['lora_B']) 
                    del pool_pick, state_dict
                else:
                    self.param_template[p].reset_parameters()
                    new_params = self.param_template[p].state_dict()
                    l_a = copy.deepcopy(new_params['lora_A'])
                    l_b = copy.deepcopy(new_params['lora_B'])
                index = self.counter
                pool_addition[p].append(ParamPair(l_a, l_b))
                index_addition[p].append(index)
            added += 1
            self.counter += 1
            self.param_pool_len += 1
            self.param_pool_total += 1

        for p in self.lora_prefixes:
            for i in range(add_count):
                self.param_pool_ref[p].append(pool_addition[p][i])
                self.param_pool[p].add(index_addition[p][i])

    def construct_tensor(self, id, state_dict):
        subset_param_dict = state_dict # copy.deepcopy(state_dict)
        subsample_index_hidden_layer = {}
        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)
            
            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"
            param_idxs = self.id_to_param[p][id]
            subsample_index_hidden_layer[p] = [param_idxs]

            construct_index_hl = torch.tensor([i for i in range(len(param_idxs))])

            A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
            B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
            update_tensor_by_update_lists_dim_0(subset_param_dict[lora_a], A_list, construct_index_hl)
            update_tensor_by_update_lists_dim_1(subset_param_dict[lora_b], B_list, construct_index_hl)
        return subset_param_dict

        pass # TODO

    def checkout(self, id):
        print("Pre-checkout active:", self.active_count, " co: ", self.checkout_count)
        #state_dict = {}
        if self.active_count - self.checkout_count < self.rank:
            self.create_new_params()

        avail_pool = self.get_avail_set()

        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)

            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"  

            #pool_list = np.array(list(self.param_pool[p])) # Sync?
            #pool_list = np.array(list(range(self.min_pool))) # Async?
            pool_list = np.array(list(avail_pool[p]))
            #all_params = list(range(self.min_pool)) # Sync
            #avail_params = [i for i in all_params if self.param_pool_inuse_flag[p][i] == 0] # Sync
            #pool_list = np.array(list(avail_params)) # Sync
            param_idxs = np.random.choice(pool_list, self.rank, replace=False)
            #self.param_pool[p].difference_update(param_idxs)

            for i in param_idxs:
                self.param_pool_checkout_flag[p][i] = 1

            self.id_to_param[p][id] = param_idxs.tolist()

        self.checkout_count += self.rank

        print("Post-checkout active:", self.active_count, " co: ", self.checkout_count)

        #self.param_pool_len -= self.rank

    def checkin(self, id, state_dict, age=None):
        print("Pre-checkin active:", self.active_count, " co: ", self.checkout_count)
        for p in self.lora_prefixes:
            param_idxs = self.id_to_param[p].pop(id, None)
            for i in param_idxs:
                self.param_pool_checkout_flag[p][i] = 0

            # Sync
            # for i in param_idxs:
            #     self.param_pool_inuse_flag[p][i] = 0

            #self.param_pool[p].update(param_idxs) # Reintroduce parameters to pool # TODO Does it matter that this is not random
            #pool_list = np.array(list(range(self.min_pool)))
            #if USE_DISTR_CHECKIN:
            #    p_distr = [i**(self.alpha * age) for i in range(self.min_pool)]
            #else:
            #    p_distr = [1.0 for i in range(self.min_pool)]
            #p_distr_norm = [p/(sum(p_distr)) for p in p_distr]

            #param_idxs = np.random.choice(pool_list, self.rank, replace=False, p=p_distr_norm)  
            #print(f"age: {age}, updated: {param_idxs}")

            lora_a, lora_b  = self.get_A_B_key(p)
            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B" 
            
            index_hidden_layer = torch.tensor(param_idxs)
            construct_index_hl = torch.tensor([i for i in range(len(index_hidden_layer))])
            part_a = partition_FC_layer_by_output_dim_0(state_dict[lora_a], construct_index_hl)
            part_b = partition_FC_layer_by_input_dim_1(state_dict[lora_b], construct_index_hl)
            for i, idx in enumerate(index_hidden_layer):
                # It's okay if we screw up the order :D
                self.param_pool_ref[p][idx].set_A(part_a[i])
                self.param_pool_ref[p][idx].set_B(part_b[i])
                #self.param_pool_ref[p][idx].checkin()

        #self.param_pool_len += self.rank
        self.checkout_count -= self.rank

        print("Post-checkin active:", self.active_count, " co: ", self.checkout_count)

    def add_client(self, id):
        p0 = [p for p in self.lora_prefixes][0]
        if len(self.param_avail_index[p0]) == 0:
            update_flag = True
            self.param_pool_len += self.rank
        else:
            update_flag = False

        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)

            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"  
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
        self.part += 1

    def generate_model_params(self, id, state_dict):
        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)

            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"    
            index_hidden_layer = torch.tensor(self.id_to_param[p][id])

            A_list = [self.param_pool[lora_a][idx] for idx in index_hidden_layer]
            B_list = [self.param_pool[lora_b][idx] for idx in index_hidden_layer]

            construct_index_hl = torch.tensor([i for i in range(len(index_hidden_layer))])

            update_tensor_by_update_lists_dim_0(state_dict[lora_a], A_list, construct_index_hl)
            update_tensor_by_update_lists_dim_1(state_dict[lora_b], B_list, construct_index_hl) 
        return state_dict 

    def update(self, id, state_dict):
        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)
            
            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B" 
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
        # Only sample from columns that are trained
        subset_param_dict = copy.deepcopy(state_dict)
        subsample_index_hidden_layer = {}
        # Debug
        #u0 = list(self.u_ids)[0]
        active_set = self.get_active_set()
        for p in self.lora_prefixes:
            lora_a, lora_b  = self.get_A_B_key(p)
            
            #lora_a = f"{p}.lora_A"
            #lora_b = f"{p}.lora_B"
            trained_params = list(active_set[p])
            #trained_params = [idx for idx, v in enumerate(self.param_pool_ref[p]) if v.active]
            #trained_params = [idx for idx in range(self.maxrank) if self.trained[p][idx]==1]
            param_idxs = np.random.choice(np.array(trained_params), rank, replace=False)    
            #param_idxs = self.id_to_param[p][u0]
            subsample_index_hidden_layer[p] = [param_idxs]

            construct_index_hl = torch.tensor([i for i in range(len(param_idxs))])

            A_list = [self.param_pool_ref[p][idx].A for idx in param_idxs]
            B_list = [self.param_pool_ref[p][idx].B for idx in param_idxs]
            update_tensor_by_update_lists_dim_0(subset_param_dict[lora_a], A_list, construct_index_hl)
            update_tensor_by_update_lists_dim_1(subset_param_dict[lora_b], B_list, construct_index_hl)
            #subset_param_dict[lora_a] = partition_FC_layer_by_output_dim_0(self.A, subsample_index_hidden_layer[p])[0]
            #subset_param_dict[lora_b] = partition_FC_layer_by_input_dim_1(self.B, subsample_index_hidden_layer[p])[0]
        return subset_param_dict, subsample_index_hidden_layer
    
    #def get_batch_statistics(self):
