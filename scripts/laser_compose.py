import os
import json
import time
import torch
import pickle
import logging
import argparse
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
from random import shuffle
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM



parser = argparse.ArgumentParser(description='Process Arguments')	

parser.add_argument('--val_set', type=float, default=0.2, help='size of the val set for each iteration')
parser.add_argument('--log_dir',
                    type=str,
                    default="./",
                    help="Place to save log")

parser.add_argument('--home_dir', type=str,
                    default="./compose",
                    help='Directory where the data is')

parser.add_argument('--dataset_dir', type=str,
                    default="counterfact.json",
                    help='Directory where the data is')

args = parser.parse_args()

# Model and Dataset Load
model_o = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16
)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
home_dir = args.home_dir
dataset_loc = args.dataset_dir
with open(dataset_loc) as f:
    d = json.load(f)

# Helper Funcs

def do_low_rank(weight, k, debug=False):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        log(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=2)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        log(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

# Helper functions for matrix update
def get_parameter(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def update_model(model, name, params):
    with torch.no_grad():
        get_parameter(model, name)[...] = params

def compute_loss(model_edit,dataset,dtpts): 
    '''
    takes a dataset and a model and returns the accuracy of the model on the dataset
    '''
    loss = []
    lossFn = torch.nn.CrossEntropyLoss()
    for i in tqdm(dtpts):
        paraphrases = [dataset[i]['requested_rewrite']['prompt'].replace('{}', d[i]['requested_rewrite']['subject'])]
        inputs = tokenizer(paraphrases[0], return_tensors="pt").to(device)
        outputs = model_edit(**inputs, labels=inputs["input_ids"])              
        output_logits = outputs.logits[:, -1, :]
        out_gt = torch.tensor(
            tokenizer(' '+dataset[i]['requested_rewrite']['target_true']['str'])["input_ids"]).long().to(device)

        error = lossFn(output_logits, out_gt)
        loss.append(float(error.detach().cpu().numpy()))
    return np.mean(loss)




def compute_acc(model_edit,dataset,dtpts): 
    '''
    takes a dataset and a model and returns the accuracy of the model on the dataset
    '''
    loss = []
    for i in tqdm(dtpts):
        paraphrases = [dataset[i]['requested_rewrite']['prompt'].replace('{}', d[i]['requested_rewrite']['subject'])]+ d[i]['paraphrase_prompts']
        for j in range(3):
            inputs = tokenizer(paraphrases[0], return_tensors="pt").to(device)
            outputs = model_edit(**inputs, labels=inputs["input_ids"])              
            output_logits = outputs.logits[:, -1, :]
            gold_answer = dataset[i]['requested_rewrite']['target_true']['str']

            probs = torch.log_softmax(outputs.logits, dim=-1).detach()

            out_gt = torch.tensor(
                tokenizer(' '+dataset[i]['requested_rewrite']['target_true']['str'])["input_ids"]).long().to(device)

            sorted_prob, sorted_indices = torch.sort(probs[0, -1, :], descending=True)
            sorted_indices = sorted_indices[:1].detach()
            decoded_tokens = tokenizer.batch_decode(sorted_indices)
            top_k_tokens = [token for token in decoded_tokens]

            is_correct = gold_answer.lower().strip() in [token.lower().strip() for token in top_k_tokens]
            if is_correct:
                loss.append(1)
            else:
                loss.append(0)
    return np.mean(loss)

lname_to_int = {'k_proj':0, 'q_proj':1, 'v_proj':2, 'out_proj':3, 'fc_in':4, 'fc_out':5}


def edit_layer(model, model_edit,reductions): 
    for n, p in tqdm(model.named_parameters()):  # for every layer
        name = n.split(".")   # if layer in one of the select ones
        save_as = ".".join(name[3:])
        if len(name) > 3 and name[-1] == "weight" and name[3]=='mlp' and name[4] == 'fc_in':
            feats_max = min(p.shape[0],p.shape[1])

            lnum = int(name[2])
            # lname = int(lname_to_int[name[4]])

            rate = reductions[0,lnum]

            if rate>0:
                print(n) 
                feats = list(range(int((1-rate)*feats_max)))

                results = pickle.load(open(f"{args.home_dir}{save_as}-{name[2].zfill(3)}.p","rb")) # Save the SVD of the matrix so it doesnt need to be computed repeatedly
                u = results[0][:,feats]
                s = torch.diag(results[1][feats])
                vt = results[2].T[feats,:]
                print("Results:", u.shape,s.shape,vt.shape,len(feats))
                new_mat = torch.matmul(torch.matmul(u,s),vt)

                update_model(model_edit, n, new_mat)

    return model_edit

def return_loss(model, model_edit,reductions,d,dtpts_to_use):
    # 1. Baseline
    model_edit = edit_layer(model, model_edit,reductions)
    model_edit.to(device)

    # 2. Validate over a randomly chosen subset of the dataset
    loss = compute_loss(model_edit,d,dtpts_to_use)

    return loss

def return_accuracy(model, model_edit,reductions,d,dtpts_to_use):
    # 1. Baseline
    model_edit = edit_layer(model, model_edit,reductions)
    model_edit.to(device)

    # 2. Validate over a randomly chosen subset of the dataset
    acc = compute_acc(model_edit,d,dtpts_to_use)

    return acc


def reduce_step(reductions,i,j):
    stop_red = 0
    # if ct==0:
        # reductions[i,j] = 0.0
    if reductions[i,j]<0.9:
        reductions[i,j]+= 0.45
    elif reductions[i,j]<0.98:
        reductions[i,j]+= 0.04
    else:
        stop_red=1
    
    return reductions,stop_red

def stacking():

    all_loss = {}

    dtpts_to_use = list(range(len(d)))
    dtpts_to_use = dtpts_to_use[:int(args.val_set*len(d))]

    model_edit = deepcopy(model_o)
    
    reductions = torch.from_numpy(np.zeros((1,28))) # Only MLP layer: 1 layer types and 28 layers

    
    accuracy_ = return_accuracy(model_o, model_edit,reductions,d,dtpts_to_use)
    all_loss['start']=accuracy_

    print(accuracy_)
    print(reductions)

    lnums = list(range(27))[::-1]

    for lnum in lnums:
        stop_red=0 # stop reducing further for this layer?

        while (accuracy_>=np.min(list(all_loss.values()))) and (stop_red==0): # till best reduction for the layer is reached do:
            model_edit = deepcopy(model_o) 
            old_red = reductions
            reductions, stop_red= reduce_step(reductions,0,lnum)
            accuracy_ = return_accuracy(model_o, model_edit,reductions,d,dtpts_to_use)
            print(0,lnum,reductions[0,lnum],accuracy_)

            all_loss[f"{lnum}-{0}-{reductions[0,lnum]}"]=accuracy_
            
        if accuracy_<np.min(list(all_loss.values())):
            reductions = old_red
    pickle.dump(reductions,open("reductions.p","wb"))
    pickle.dump(all_loss,open("all_acc.p","wb"))
    return reductions

if __name__ == '__main__':
    stacking()