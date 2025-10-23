import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random

os.environ["WANDB_DISABLED"] = "true"

from termcolor import colored

from transformers import LlamaConfig, MistralConfig, AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser, Namespace
import torch.distributed as dist
import datetime


class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process

def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}" 
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))
    
    return DistConfig(is_distributed, rank, world_size, device, master_process)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        
        context_length = input.input_ids.shape[-1]
        
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shadowkv_datalen", type=int, default=8*1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=256)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)
    p.add_argument("--e", action='store_true', default=False)

    return p.parse_args()

if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    dist_config = init_dist()

    args = parse_args()
    model_full_name = args.model_name
    batch_size = args.batch_size
    shadowkv_datalen = args.shadowkv_datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference


    model_name = model_full_name.split("/")[-1]

    
    if 'llama' in model_name.lower():
        config = LlamaConfig.from_pretrained(model_full_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_full_name, 
        #                                     use_fast=False, 
        #                                     trust_remote_code=True, 
        #                                     tokenizer_type='llama')
        tokenizer = AutoTokenizer.from_pretrained(model_full_name, tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)

    else:
        raise NotImplementedError
    

    # Load model directly
    model = AutoModelForCausalLM.from_pretrained(model_full_name).to(dist_config.device)
    model.eval()

    max_length = model2maxlen[model_name]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # based on SpeCache Appendix C. Full Results on LongBench
        datasets = ["narrativeqa", 
                    "qasper",
                    "multifieldqa_en",
                    "hotpotqa",
                    "2wikimqa",
                    "musique",
                    "gov_report", 
                    "qmsum",
                    "multi_news",
                    "trec",
                    "triviaqa",
                    "samsum",
                    "passage_retrieval_en", 
                    "lcc",
                    "repobench-p"]



    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # predict on each dataset

    os.makedirs("longbench_pred", exist_ok=True)
    
    if args.e:
        if not os.path.exists("longbench_pred_e"):
            os.makedirs("longbench_pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}"):
                os.makedirs(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}")
            out_path = f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            os.makedirs(f"longbench_pred/{model_name}_{max_length}_{args.method}", exist_ok=True)
            out_path = f"longbench_pred/{model_name}_{max_length}_{args.method}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, dist_config.device, model_name)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')