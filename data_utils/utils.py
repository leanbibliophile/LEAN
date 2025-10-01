from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import os
import functools
import loralib as lora 
import random
from calflops import calculate_flops
import copy
from tqdm import tqdm
import torch.nn as nn

import pandas as pd

from peft import set_peft_model_state_dict, get_peft_model_state_dict

ds_transforms = {
    "train": transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
}

def get_transforms(mode):
    return ds_transforms[mode]

def read_traintest_dir(root_path):
    file_list = []
    label_list = sorted(os.listdir(root_path))
    label_index = {l: idx for idx, l in enumerate(label_list)}
    label_count = [0 for i in label_index]
    labels = []
    for l in label_list:
        label_dir = os.path.join(root_path, l)
        files = os.listdir(label_dir)
        file_list.extend((os.path.join(label_dir, f) for f in files))
        labels.extend((label_dir for f in files))
        label_count[label_index[l]]=len(files)
    return file_list, labels, label_index, label_count

def get_label_count(file_list, label_index):
    label_count = [0 for i in label_index]
    for f in file_list:
        label = label_index[f.split("/")[-2]] 
        label_count[label]+=1
    return label_count


class ImageDataset(Dataset):
    def __init__(self, file_list, label_index, transform=None, images_list=None, dtype=torch.float32):
        self.file_list = file_list
        self.transform = transform
        
        self.label_index = label_index
        self.classes = label_index.values()
        self.targets = [label_index[f.split("/")[-2]] for f in file_list]

        self.type = dtype

        #if images_list is None:
        #    self.images = [Image.open(f).convert('RGB') for f in self.file_list]
        #else:
        #    self.images = images_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        # img = self.images[idx]
        if self.transform is not None:
            img_transformed = self.transform(img).to(self.type)
        else:
            img_transformed = img

        label = self.targets[idx]
        return img_transformed, label
    
class ClientDataset(Dataset):
    # Dataset sample at client
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample, label = self.dataset[self.idxs[item]]
        return sample, label

class ClientDatasetLLM(Dataset):
    # Dataset sample at client
    def __init__(self, dataset, idxs, eval=False):
        self.dataset = dataset
        self.idxs = idxs
        self.eval = eval
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]
        #if not self.eval:
        #    id, mask, label = self.dataset[self.idxs[item]]
        #    return id, mask, label
        #else:
        #    id, mask, label = self.dataset[self.idxs[item]] 
        #    return id, mask

class ActivationExtractor:
    def __init__(self):
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def get_output(self, name):
        return self.activation[name]

# Recursive get attribute
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def get_lora_layers(model):
    lora_prefixes = {}
    for k in model.state_dict().keys():
        if "lora_A" in k:
            index = ".".join(k.split(".")[:-1])
            if index not in lora_prefixes:
                lora_prefixes[index] = True
    lora_prefixes = list(lora_prefixes.keys())

    return lora_prefixes

# This is too slow
def lora_hooks(model):
    ae = ActivationExtractor()
    layers = get_lora_layers(model)

    for k in layers:
        layers.append(k)
        p = rgetattr(model, k)
        p.register_forward_hook(ae.get_activation(k))
        print(f"Hooked {p}")

    print("Done hooking")
    return ae, layers

def get_statistics(tensor, dim=0):
    first_moment = torch.mean(tensor, dim=dim)
    second_moment = torch.mean(torch.square(tensor), dim=dim)
    return first_moment, second_moment

def shift_batch_stats(batch, mean, var, dim=0):
    batch = batch.to(torch.float32).detach()
    batch_size = batch.size(dim)

    batch_mean = torch.mean(batch, dim=dim)

    #batch_var= torch.var(batch, dim=dim)
    # Maybe mask out the zero batch stddev parts
    sigma = torch.sqrt(var) 
    batch_sigma = torch.std(batch, dim=dim) 
    # DEV
    batch_sigma_mask = (batch_sigma!=0).to(torch.float32)

    rescale = torch.div(sigma, batch_sigma)
    bias = torch.sub(mean, torch.mul(batch_mean, rescale))

    batch_split = torch.tensor_split(batch, batch_size, dim=dim)
    batch_stack = []
    for x in batch_split:
        x = torch.squeeze(x)
        rescale_x = torch.mul(x, rescale)
        rescale_x = torch.add(rescale_x, bias)
        rescale_x = torch.nan_to_num(rescale_x, nan=9e6) # Replace nans
        rescale_x = torch.mul(rescale_x, batch_sigma_mask)
        batch_mask_complement = 1 - batch_sigma_mask
        rescale_x = torch.add(rescale_x, torch.mul(x, batch_mask_complement))
        batch_stack.append(rescale_x)

    new_batch = torch.stack(batch_stack, dim=dim)
    new_batch = new_batch.to(torch.float32)
    return new_batch

def partition_dataset(dataloader, clients=1):
    n = len(dataloader)
    q = n // clients
    r = n % clients
    samples = [q + 1 if i < r else q for i in range(n)] # Sample roughly evenly
    dict_users = {}
    partition_list = [i for i in range(n)]
    random.shuffle(partition_list)
    pointer = 0
    for i in range(clients):
        endp = pointer + samples[i] 
        dict_users[i] = [v for v in partition_list[pointer:endp]] 
        pointer = endp
    return dict_users

def get_input_shape(dataset, batch_size):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    for data, labels in loader:
        break
    return tuple(data.shape)

import math
def get_size_layer(tensor, dtype=None):
    shape = tensor.shape
    if dtype is None:
        tensor_dtype_size = tensor.dtype.itemsize
    else:
        tensor_dtype_size = dtype.itemsize
    byte_size = tensor_dtype_size * math.prod(shape) 

    return byte_size

def get_size_transfer_model(state_dict, dtype=None):
    byte_total = 0
    for layer, tensor in state_dict.items():
        byte_total += get_size_layer(tensor, dtype)
    
    return byte_total

def evaluate(model_class, model_name, num_classes, device, dataloader, pretrained_state_dict, fl_instance, rank_list, use_stats):
    layer_statistics = fl_instance.aggregate_statistics() # I shouldn't need to shuffle layer statistics?

    #default_layer = list(fl_instance.param_library.param_pool_ref.keys())[0]
    #trained_count = len([i for i in fl_instance.param_library.param_pool_ref[default_layer] if i.trained])
    #print(f"library size: {trained_count}")
    output = []
    if use_stats:
        use_stats_list = [True, False] # [True]
    else: 
        use_stats_list = [False]
    for enable_stats in use_stats_list:
        print(f"enable stats: {enable_stats}")
        for eval_lora_rank in rank_list:
            e = eval_lora_rank
            #if e > trained_count:
            #    break
            eval_model = model_class(model_name, num_classes=num_classes, pretrained=False, lora_rank=e, lean=True, lora_alpha=1, enable_stats=enable_stats)
            eval_model.load_state_dict(pretrained_state_dict, strict=False)
            eval_model_dict = eval_model.state_dict()
            eval_model_dict, eval_index = fl_instance.param_library.get_rank_subset(e, eval_model_dict)
            eval_model_dict = eval_model_dict | layer_statistics

            #model.load_state_dict(central_model_dict, strict=False)
            eval_model.load_state_dict(eval_model_dict, strict=False)

            eval_model.train() # ????
            eval_model = eval_model.to(device)

            # Only evaluate every 20 rounds to speed things up
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                counts = 0
                for data, label in tqdm(dataloader):
                    data = data.to(device)
                    label = label.to(device)
                    
                    val_output,_ = eval_model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(dataloader)
                    epoch_val_loss += val_loss / len(dataloader)

                    val_loss = criterion(val_output, label)
            
                output.append({
                    "enable_stats": enable_stats,
                    "rank": e,
                    "accuracy": epoch_val_accuracy,
                    "loss": epoch_val_loss
                })
                #log_str = f"Eval Epoch accuracy (use stats: {enable_stats}) - rank {e}: {epoch_val_accuracy}, Epoch loss: {epoch_val_loss}"
                #print(log_str)
                #with open(log_path, "a") as log_file:
                #    log_file.write(log_str + "\n")
                #wandb.log({"Test acc": epoch_val_accuracy, "Eval rank": e})
    return output

class Logger:
    HEADER = ["accuracy", "bytes_transfer","flops"]

    def __init__(self, exp_settings, eval_ranks, output_dir):
        self.clients = exp_settings["population"]
        self.diric = exp_settings["diric"]
        self.rank = exp_settings["train_rank"]
        self.localep = exp_settings["loc_ep"]
        self.alpha = exp_settings["lora_alpha"]
        self.lr = exp_settings["learning_rate"]
        self.use_prox = exp_settings["use_prox"]
        self.prox = exp_settings["prox"]
        self.alg = exp_settings["algorithm"]
        self.dataset = exp_settings["dataset"]

        self.out_dir = output_dir
        self.eval_ranks = eval_ranks
        self.log_entries = {r: pd.DataFrame(columns=self.HEADER) for r in eval_ranks}

    def generate_fname(self, rank):
        log_str = f"{self.alg}_{self.dataset}_pop{self.clients}_dir{self.diric}_loc{self.localep}_alpha{self.alpha}_lr{self.lr}_evalrank{rank}"

        if self.use_prox:
            log_str = f"{log_str}_mu{self.prox}"
        else: 
            log_str = f"{log_str}"

        temp_log_str = os.path.join(self.out_dir, f"{log_str}.csv")
        if os.path.exists(temp_log_str):
            log_str = os.path.join(self.out_dir, f"{log_str}_temp.csv") # Generate a copy to avoid 
        else:
            log_str = temp_log_str

        return log_str
    
    def record(self, rank, accuracy, bytes_transfer, flops):
        self.log_entries[rank] = pd.concat([self.log_entries[rank], pd.DataFrame([[accuracy, bytes_transfer, flops]], columns=self.HEADER)])

    def write(self, rank, fname=None):
        if fname is None:
            fname = self.generate_fname(rank)

        self.log_entries[rank].to_csv(fname, columns=self.HEADER, index=False)


cpu_device = torch.device('cpu')

QWEN_FULLRANK = 1024

def adhoc_update(model, state_dict):
    layers = model.base_model.model.model.layers
    blocks = len(layers)
    for i in range(blocks):
        base_layer_q = layers[i].self_attn.q_proj.base_layer
        base_layer_v = layers[i].self_attn.v_proj.base_layer
        
        key = f"base_model.model.model.layers.{i}.self_attn"
        if f"{key}.q_proj.base_layer.weight" in state_dict:
            base_layer_q.weight = nn.Parameter(state_dict[f"{key}.q_proj.base_layer.weight"].to(torch.bfloat16))
        if f"{key}.v_proj.base_layer.weight" in state_dict:
            base_layer_v.weight = nn.Parameter(state_dict[f"{key}.v_proj.base_layer.weight"].to(torch.bfloat16))

def evaluate_llm(base_model, gt_set, tokenizer, device, dataloader, fl_instance, rank_list, flora=False):
    from qwen_chem_utils import eval, get_lora_model
    #layer_statistics = fl_instance.aggregate_statistics() # I shouldn't need to shuffle layer statistics?

    #default_layer = list(fl_instance.param_library.param_pool_ref.keys())[0]
    #trained_count = len([i for i in fl_instance.param_library.param_pool_ref[default_layer] if i.trained])
    #print(f"library size: {trained_count}")
    output = []
    
    for eval_lora_rank in rank_list:
        if flora:
            e = QWEN_FULLRANK
        else:
            e = eval_lora_rank
        #if e > trained_count:
        #    break
        eval_model = copy.deepcopy(base_model)
        eval_model = eval_model.to(cpu_device)
        eval_model = get_lora_model(eval_model, e, inference=True)
        if flora:
            eval_model_dict = eval_model.state_dict()             
            eval_model_dict = fl_instance.param_library.flora_llm_sd(eval_model_dict)
            eval_model.merge_adapter()
            eval_model.load_state_dict(eval_model_dict, strict=False)
            adhoc_update(eval_model, eval_model_dict)
        else:
            eval_model_dict = get_peft_model_state_dict(eval_model)
            eval_model_dict, eval_index = fl_instance.param_library.get_rank_subset(e, eval_model_dict)
            set_peft_model_state_dict(eval_model, eval_model_dict)

        eval_model.eval() # ????
        eval_model = eval_model.to(device)

        accuracy = eval(eval_model, dataloader, gt_set, tokenizer, device)
       
        output.append({
            "rank": e,
            "accuracy": accuracy,
        })
        
    del eval_model
    return output

