import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
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

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k")
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    p.add_argument("--sparse_budget", default=2048)
    p.add_argument("--rank", type=int, default=160)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--minference", action='store_true', default=False)
    #============================================================
   

    return p.parse_args()


"""
Example usage:
OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 1 test/debug_specache.py --datalen 131072 --method specache --dataset_name "ruler/niah_single_2" --model_name "/mnt/cephfs/models/meta-llama/Llama-3.1-8B-Instruct"
"""
if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    num_samples = args.num_samples
    datalen = args.datalen # 128*1024 = The length of the context
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference

    dist_config = init_dist()
    
    from evaluator_specache import Evaluator
    from specache import choose_model_class
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config)
    
    
    LLM = choose_model_class(model_name)

    llm = LLM(model_name=model_name, batch_size=batch_size, device=dist_config.device, max_length=datalen+2048, attn_mode=args.method, dtype=dtype, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size, minference=minference)

    if dist_config.master_process:
        llm.print_kv_stats()

    dataset = Dataset("ruler/niah_single_2", llm.tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size)
    if args.method != 'specache':
        output_path = f"debug/{model_name.split('/')[-1]}/ruler/niah_single_2_{datalen}_{args.method}_{sparse_budget}_{rank}_{chunk_size}.jsonl"
    else:
        output_path = f"debug/{model_name.split('/')[-1]}/ruler/niah_single_2_{datalen}_{args.method}_minf{minference}.jsonl"
    evaluator.test(llm, dataset, output_path, args.method)
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

