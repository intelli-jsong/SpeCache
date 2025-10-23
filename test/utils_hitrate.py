from argparse import ArgumentParser, Namespace
import torch
import os
from pathlib import Path

def analysis_per_sample(sample_no, base_path, top_k):
    sample_file = f"{base_path}/sample_{sample_no}.pt"
    
    sample_result = torch.load(sample_file)

    # calculate Speculative Token Exact Hit Rate
    gt_token_ids = torch.tensor(sample_result['generated_ids'])[1:]
    spec_token_ids = torch.tensor(sample_result['spec_token_ids'])[:-1]
    
    # sprint(f"Sample {sample_no}:")
    # print(f"> Output: {sample_result['decoded_output']}")
    # print(f"> GT Token IDs: {gt_token_ids.tolist()}")
    # print(f"> Spec Token IDs: {spec_token_ids.tolist()}")
    # print(sample_result['top_k_hitrates'][0]/64)
    # print(sample_result['top_k_hitrates'][15]/64)
    # print(sample_result['top_k_hitrates'][30]/64)


    exact_hits = (gt_token_ids == spec_token_ids).sum().item()
    exact_hit_rate = exact_hits / len(gt_token_ids)
    
    # calculate Speculative Token Top-K KV Cache Hit Rate
    top_k_hitrates = sample_result['top_k_hitrates']
    for layer_idx in list(top_k_hitrates.keys()):
        top_k_hitrates[layer_idx] = torch.mean(top_k_hitrates[layer_idx], dim=1) / top_k  # (num_heads,)
    
    return exact_hit_rate, top_k_hitrates
    
def calc_topk_hitrates_mean(top_k_hitrates):

    num_layers = len(top_k_hitrates)
    num_head = top_k_hitrates[0].shape[0]

    accum = 0
    for layer_idx in list(top_k_hitrates.keys()):
        accum += torch.sum(top_k_hitrates[layer_idx]).item()
    
    return accum / (num_layers * num_head)

def analysis_total_dataset(base_path, top_k):

    sample_files = [f for f in os.listdir(base_path) if f.startswith("sample_") and f.endswith(".pt")]
    
    exact_hit_accum = 0
    topk_hit_accum = 0

    for sample_f in sample_files:
        sample_no = int(sample_f.split('_')[1].split('.')[0])
        exact_hit_rate, top_k_hitrates = analysis_per_sample(sample_no, base_path, top_k)

        exact_hit_accum += exact_hit_rate

        topk_hitrates_mean = calc_topk_hitrates_mean(top_k_hitrates)
        topk_hit_accum += topk_hitrates_mean

        print(f"\nSample {sample_no}: Exact Hit Rate = {exact_hit_rate:.2f}, Top-{top_k} Hit Rates = {topk_hitrates_mean:.2f}")
        
    num_samples = len(sample_files)
    print("==============================================================")
    print(base_path)
    print(f"> Overall Exact Hit Rate = {exact_hit_accum / num_samples:.2f}")
    print(f"> Overall Top-{top_k} Hit Rate = {topk_hit_accum / num_samples:.2f}")  

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="Meta-Llama-3-8B-Instruct")
    p.add_argument("--specache_bit", default=2)
    p.add_argument("--dataset", type=str, default="gov_report")
    p.add_argument("--top_k", type=int, default=64)

    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()

    base_path = f"{Path(__file__).parent.parent}/longbench_specache_hitrate/{args.model_name}_{args.specache_bit}bit_{args.dataset}"
    analysis_total_dataset(base_path, args.top_k)