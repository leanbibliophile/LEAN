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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import loralib as lora

# hf
#from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import get_scheduler
from evaluate import load

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm

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

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors

from peft import LoraModel, LoraConfig, get_peft_model
#from custom_peft import LoraModel, LoraConfig, get_peft_model

seed = 926

MAX_LENGTH = 128
QWEN_MAX_LENGTH=1024 # This is only for training, set to 128 in evaluation
NUM_EXAMPLES = 0 # Number for ICL. I disabled this


# Cache for dataset loading and processing to save time
cache_path = "/ws/fs_mount/chem_ds/ChemLLMBench/data/reaction_prediction/dataset_cache_small_noicl.pickle" 
USE_CACHE = True 
SAVE_CACHE = True

FINETUNE = False # Finetune versus use existing model
VALID_SAMPLE=500 # Sample size for mid training validation
DO_VALID = False # Mid training validation, kinda slow, otherwise does validation at end of training
USE_LOAD = True # Load fine-tuned model or use pre-trained model

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

model_name = "qwen"

metric = load("accuracy")

#chem_df_path = "/ws/fs_mount/chem_ds/ChemLLMBench/data/reaction_prediction/uspto_mixed.pickle"
chem_df_path = "/ws/fs_mount/chem_ds/ChemLLMBench/data/reaction_prediction/uspto_50.pickle"
small_df_path  =  "/ws/fs_mount/chem_ds/ChemLLMBench/data/reaction_prediction/small.pickle" 

def generate_small_df(df):
    train = df[df['set'] == 'train'].head(5000)
    valid = df[df['set'] == 'valid'].head(5000)
    test = df[df['set'] == 'test'].head(5000) 

    small_df = pd.concat([train, valid, test], axis=0)

    small_df.to_pickle(small_df_path)

def get_chem_df(path):
    df = pd.read_pickle(path)
    generate_small_df(df)

    df['reactants_smiles'] = df['reactants_mol'].apply(lambda x: Chem.MolToSmiles(x))
    df['products_smiles'] = df['products_mol'].apply(lambda x: Chem.MolToSmiles(x))

    df = df.drop('reactants_mol', axis=1)
    df = df.drop('products_mol', axis=1)

    select_columns = ['reactants_smiles', 'products_smiles']

    train = Dataset.from_pandas(df[df['set'] == 'train'])
    train = train.select_columns(select_columns)
    valid = Dataset.from_pandas(df[df['set'] == 'valid'])
    valid = valid.select_columns(select_columns)
    test = Dataset.from_pandas(df[df['set'] == 'test'])
    test = test.select_columns(select_columns)
    
    return train, valid, test

def get_scaffold_fp(x):
    mol = Chem.MolFromSmiles(x['reactants_smiles'])
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
    return {
        "scaffold_fp": scaffold_fp
    }

def model_selector(model_name, version=None):
    if version is None:
        version = "3-0.6B"
    if model_name == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen{version}", padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/Qwen{version}", torch_dtype=torch.bfloat16)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    return model, tokenizer, data_collator

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    generator = np.random.default_rng(seed)
    return generator

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def check_smiles(str1, gt):
    try:
        #str1 = str1.strip(punct_exclude_endbrack) # Clean leading/trailing punctuation?
        mol = Chem.MolFromSmiles(str1)
        if mol is None:
            return False
        pred = Chem.MolToSmiles(mol)
    except Exception:
        return False

    pred_list = pred.split(".")
    if gt in pred_list:
        return True 
    return False

def parse_response(responses, gt_batch):
    if type(responses) is str or len(gt_batch) == 1:
        responses = [responses]
    res = []
    for r, gt in zip(responses, gt_batch):
        flag = False
        if type(r) is str:
            if check_smiles(r.strip(), gt.strip()):
                flag = True
        else:
            for s in r:
                if check_smiles(s.strip(), gt.strip()):
                    flag = True
        res.append(flag)
    
    return res

def eval(eval_model, eval_dataloader, gt_set, tokenizer, eval_device):
    eval_model.eval()
    # Skip dataloader for this for now
    gt_pointer = 0
    total_correct = 0.0
    total_samples = 0
    bar = tqdm(eval_dataloader)
    for batch in bar:
        bs = len(batch["input_ids"])
        total_samples += bs
        next_gt_pointer = gt_pointer + bs
        gt_batch = gt_set[gt_pointer:next_gt_pointer]
        gt_pointer = next_gt_pointer

        batch = {k: v.to(eval_device) if k != "labels" else v for k, v in batch.items()}
        with torch.no_grad():
            GENERATE = True
            if GENERATE:
                generated_ids = eval_model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=MAX_LENGTH,
                    #temperature=0.1,
                    #num_return_sequences=5,
                    #num_beams=5
                )
                offset = [len(input_id) for input_id in batch["input_ids"]]
                generated_ids_1 = [res[off:] for off,res in zip(offset, generated_ids)]
                response = tokenizer.batch_decode(generated_ids_1, skip_special_tokens=True)
                #print(response)
                corrects = parse_response(response, gt_batch)
            else:
                outputs = eval_model(**batch)

        comp = sum(corrects)
        total_correct += comp

        bar.set_postfix_str(f"Accuracy: {total_correct/total_samples}")

    score = total_correct / total_samples

    print(f"Acc: {score}")

    return score

    # How to do this for HF models?
    # Compute flop usage
    #train_input_shape = get_input_shape(train_data, settings["batch_size"])
    #base_batch_flops, _, _ = calculate_flops(flop_model, train_input_shape, include_backPropagation=True, output_as_string=False, output_precision=4)

def top_n_scaffold_similar_molecules(target_smiles, molecule_scaffold_list, molecule_smiles_list, n=5):
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)

    similarities = []

    for idx, scaffold_fp in enumerate(molecule_scaffold_list):
        try:
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            similarities.append((idx, tanimoto_similarity))
        except Exception as e:
            print(e)
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_similar_molecules = similarities[:n]

    return [molecule_smiles_list[i[0]] for i in top_n_similar_molecules]

def create_prompt_icl(input_text, examples):
    reactant = input_text["reactants_smiles"]
    prompt = "You are an expert chemist. Given the reactants SMILES, your task is to predict the main product SMILES using your experienced chemical Reaction Prediction knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the product. The input contains the reactants and reagents which are split by '.'. The product smiles must be valid and chemically reasonable. \n"
    input = "" 
    for example in examples:
        input += f"Reactants+Reagents: {example[0]}\nProducts: {example[1]}\n\n"
    input += f"Reactants+Reagents: {reactant}\nProducts:"
    return {
        "prompt": prompt,
        "input": input
    }

def filter_sim(scaffold, sim):
    return [
        [r,p] 
        for r, p in zip(scaffold["reactants_smiles"], scaffold["products_smiles"])
        if r in sim
    ]

def process_row_example(entry, scaffolds, generator, sample_num=5):
    reactant = entry['reactants_smiles']
    product = entry['products_smiles']

    # Sim sampling
    # Does this need optimizing TODO
    #sim = top_n_scaffold_similar_molecules(reactant, scaffolds["scaffolds"], list(scaffolds['reactants_smiles']), n=sample_num)
    #chunk = filter_sim(scaffolds, sim)
    
    # Random sampling
    n = len(scaffolds["scaffolds"])
    idxs = generator.choice(n, size=sample_num, replace=False)
    chunk = [[scaffolds["reactants_smiles"][i], scaffolds["products_smiles"][i]]for i in idxs]
    
    examples = chunk #list(zip(chunk["reactants_smiles"], chunk["products_smiles"]))

    icl_prompt = create_prompt_icl(entry, examples)

    return {
        "prompt": icl_prompt["prompt"],
        "input": icl_prompt["input"],
        "output": product
    }

def process_row_tokenize(entry, tokenizer, train=True):
    input_ids, attention_mask, labels = [], [], []
    EOS_TOKEN = tokenizer.eos_token 

    instruction = tokenizer(
        f"<|im_start|>system\n{entry['prompt']}<|im_end|>\n<|im_start|>user\n{entry['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False  # Do not add special tokens (e.g., BOS, EOS) automatically.
    )

    response = tokenizer(
        f"{entry['output']}",
        add_special_tokens=False  # Do not add special tokens automatically.
    )

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]

    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > QWEN_MAX_LENGTH:
        input_ids = input_ids[:QWEN_MAX_LENGTH]
        attention_mask = attention_mask[:QWEN_MAX_LENGTH]
        labels = labels[:QWEN_MAX_LENGTH]

    if not train:
        labels = f"{entry['output']}"
        msg = [
            {'role': 'system', 'content': entry['prompt']},  # System instruction.
            {'role': 'user', 'content': entry['input']}  # User input.
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        if len(input_ids) > MAX_LENGTH: # Redundant, not clean code TODO
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

    return {
        "input_ids": input_ids,  # Token IDs for the input sequence.
        "attention_mask": attention_mask,  # Attention mask for the input sequence.
        "labels": labels  # Labels for the model (with instruction part masked).
    }

from rdkit import RDLogger            
RDLogger.DisableLog('rdApp.*')        

def fetch_dataset(use_cache, generator=None):
    if not use_cache:
        # Dataset 
        print("Get DF")
        #train, valid, test = get_chem_df(small_df_path) # Use small subset for debugging
        train, valid, test = get_chem_df(chem_df_path)
        print("Get scaffold fp")
        train_sample = train.shuffle(seed=seed)
        scaffold = {
            "scaffolds": [get_scaffold_fp(entry)["scaffold_fp"] for entry in tqdm(train_sample)],
            "reactants_smiles": train_sample["reactants_smiles"],
            "products_smiles": train_sample["products_smiles"]
        }
        print("Map scaffold")

        if generator is None:
            generator = seed_everything(seed)

        train = train.map((lambda entry: process_row_example(entry, scaffold, generator, sample_num=NUM_EXAMPLES)))
        valid = valid.map((lambda entry: process_row_example(entry, scaffold, generator, sample_num=NUM_EXAMPLES)))
        test = test.map((lambda entry: process_row_example(entry, scaffold, generator, sample_num=NUM_EXAMPLES)))

        remove_columns = train.column_names

        print("Tokenize")
        train = train.map((lambda entry: process_row_tokenize(entry, tokenizer, train=True)), remove_columns=remove_columns)
        valid = valid.map((lambda entry: process_row_tokenize(entry, tokenizer, train=False)))
        test = test.map((lambda entry: process_row_tokenize(entry, tokenizer, train=False)))

        cache_ds = {
            "train": train,
            "valid": valid,
            "test": test
        }

        if SAVE_CACHE: 
            with open(cache_path, "wb") as cache_f:
                pickle.dump(cache_ds, cache_f)

    else:
        with open(cache_path, "rb") as cache_f:
            cache_ds = pickle.load(cache_f)
        train = cache_ds["train"]
        keep_columns = train.column_names
        valid = cache_ds["valid"]
        remove_columns = [c for c in valid.column_names if c not in keep_columns]
        test = cache_ds["test"]

    valid_gt = valid["labels"]
    test_gt = test["labels"]
    valid = valid.remove_columns(remove_columns)
    valid = valid.remove_columns("labels")
    test = test.remove_columns(remove_columns)
    test = test.remove_columns("labels")

    return train, valid, test, test_gt, valid_gt

def get_lora_model(model, rank, inference=False):
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        inference_mode=inference
    )

    lora_model = get_peft_model(model, config) #LoraModel(model, config, "default")

    return lora_model

if __name__ == "__main__":

    cpu_device = torch.device('cpu')
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu') # Device for eval
    
    # Model
    model, tokenizer, collator = model_selector(model_name)

    train, valid, test, test_gt, valid_gt = fetch_dataset(USE_CACHE)

    # Test

    #train_tokenize, _ = tokenize_dataset(train, tokenizer)
    #valid_tokenize, valid_gt = tokenize_dataset(valid, tokenizer, train=False)
    #test_tokenize, test_gt = tokenize_dataset(test, tokenizer, train=False)

    lr = 3e-4 # settings['lr']

    # Do we need collator?
    train_dataloader = DataLoader(
        train, shuffle=True, batch_size=1, collate_fn=collator
    )
    valid_dataloader = DataLoader(
        valid, batch_size=2, collate_fn=collator
    )
    eval_dataloader = DataLoader(
        test, batch_size=128, collate_fn=collator, pin_memory=True
    )

    central_lora_rank = 64

    # Lora model
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=central_lora_rank,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        inference_mode=False
    )

    lora_model = get_peft_model(model, config) #LoraModel(model, config, "default")

    print_trainable_parameters(lora_model)

    optimizer = AdamW(params=lora_model.parameters(), lr=lr)

    # Demo scheduler TODO Change this?
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
 
    lora_model = lora_model.to(device)

    DISTR = False

    # Training
    if FINETUNE:
        if not DISTR:
            lora_model.train()
            for epoch in range(num_epochs):
                bar = tqdm(train_dataloader, desc=f"Training ep {epoch}")
                counter = 0
                for batch in bar:
                    counter += 1
                    if DO_VALID and counter % VALID_SAMPLE == VALID_SAMPLE-1:
                        #valid_sample = valid.shuffle().select(range(VALID_SAMPLE))
                        #valid_sample_loader = DataLoader(
                        #    valid_sample, batch_size=1, collate_fn=collator
                        #)
                        score = eval(lora_model, valid_dataloader, valid_gt, tokenizer) 
                        #score = eval(lora_model, valid_sample_loader, valid_gt, tokenizer)
                        print(f"valid acc: {score}")
                        lora_model.train()

                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = lora_model(**batch)
                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    batch_loss = loss
                    bar.set_postfix_str(f"loss: {batch_loss}")
                    #print(f"loss: {batch_loss}")

            if not DO_VALID:
                score = eval(lora_model, valid_dataloader, valid_gt, tokenizer) 
                #score = eval(lora_model, valid_sample_loader, valid_gt, tokenizer)
                print(f"valid acc: {score}")
                lora_model.train() 

        else:
            ws = torch.cuda.device_count()
            #sampler = DistributedSampler(train, seed=926)
            #mp.spawn(distr_main, nprocs=ws, args=(ws, lora_model, optimizer, lr_scheduler, train, collator, num_epochs))
            
        # Save model
        savepath = "/ws/fs_mount/lora_lib_save"
        torch.save(lora_model.to(cpu_device).state_dict(), f"{savepath}/qwen_chem_dryrun_noicl.pt")

    # Eval
    if USE_LOAD:
        loadpath = "/ws/fs_mount/lora_lib_save/qwen_chem_dryrun_noicl.pt"
        chkpt = torch.load(loadpath)
        lora_model.load_state_dict(chkpt)
        MAX_LENGTH=128 # Change sequence length to 128 for evaluation
    else:
        lora_model = model
        MAX_LENGTH=128
    # Otherwise use whatever is preloaded or trained here

    metric = load("accuracy")

    lora_model = lora_model.to(device)
    lora_model.eval()
    # Skip dataloader for this for now
    score = eval(lora_model, eval_dataloader, test_gt, tokenizer) 
    print(f"Eval acc: {score}")