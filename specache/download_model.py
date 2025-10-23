from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

def load_model_tokenizer(model_repo, local_model_dir):
    local_cache_dir = os.path.join(local_model_dir, model_repo)

    if not os.path.isdir(local_cache_dir):
        os.makedirs(local_cache_dir)
        print(f">> '{local_cache_dir}' does not exist. downloading model from huggingface hub...")
        snapshot_download(repo_id=model_repo, local_dir=local_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(local_cache_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_cache_dir, local_files_only=True)

    return tokenizer, model

def generate_answer(tokenizer, model, question):
    # Tokenize the input question
    inputs = tokenizer.encode(question, return_tensors='pt')

    # Generate the model's response
    outputs = model.generate(inputs, max_length=256, num_return_sequences=1)

    # Decode the generated response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def print_model(model_repo, local_model_dir):
    tokenizer, model = load_model_tokenizer(model_repo, local_model_dir)
    print(model)


# python specache/download_model.py
if __name__ == "__main__":
    model_repo = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_model_dir = '/mnt/cephfs/models'
    print_model(model_repo, local_model_dir)