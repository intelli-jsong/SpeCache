################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# Base LLM class

import torch
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm

from flash_attn import flash_attn_with_kvcache
from transformers.models.llama.modeling_llama import repeat_kv
from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU, SpeCacheKVCache
import yaml
from pathlib import Path


top_k_hitrates = torch.zeros(0) 

class LLM:
    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"

    def init_kv_cache(self, sparse_budget: int, rank: int, chunk_size: int, config):
        if self.attn_mode == 'full':
            self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, rank=rank, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'specache':
            config_kv_path = Path(__file__).parent / self.specache_config_file
            with open(config_kv_path, 'r', encoding='utf-8') as file:
                config_kv_args = yaml.safe_load(file)
            
            self.kv_cache = SpeCacheKVCache(config, 
                                            max_length=self.max_length, 
                                            device=self.device, 
                                            dtype=self.dtype, 
                                            batch_size=self.batch_size,
                                            quant_k_bit=config_kv_args['quant_k_bit'],
                                            quant_k_mode=config_kv_args['quant_k_mode'],
                                            quant_v_bit=config_kv_args['quant_v_bit'],
                                            quant_v_mode=config_kv_args['quant_v_mode'],
                                            do_specache_quant=config_kv_args['do_specache_quant'],
                                            quant_group_size=config_kv_args['quant_group_size'],
                                            quant_resuidual=config_kv_args['quant_resuidual'],
                                            arg_top_k=config_kv_args['arg_top_k'],
                                            cpu_mode=config_kv_args['cpu_mode'],
                                        )    
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)
       
        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        # 이전 KV cache 정보를 보존하면서 새로운 입력을 추가
        logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
        return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):
        
        """
        hidden_states.shape :
            (bsz=1, seq_len, hiddden_dim)
        """

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,           
            self.num_key_value_heads, 
            self.head_dim  
        )
        """
        query_states.shape:
            (bsz, seq_len, num_heads*head_dim)
        key_states.shape:
            (bsz, seq_len, num_key_value_heads*head_dim)
        value_states.shape:
            (bsz, num_key_value_heads, seq_len, head_dim)
        """
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU):

            if q_len > 4*1024: # prefill
                # svd unrope key and save
                self.kv_cache.get_svd(key_states, layer_idx=layer_idx)
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                """
                query_states.shape:
                    (bsz, num_heads, seq_len, head_dim)
                key_states.shape:
                    (bsz, num_key_value_heads, seq_len, head_dim)
                """
                
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                """
                prefill 후 GPU Mem. 에 저장되는 것들 :
                - self.{k/v}_cache_buffer[layer_idx]    : local_chunk들에 대한 states + rest states + outlier chunk들에 대한 states 저장
                    * 참고 : key outlier chunk들 속 state들은 RoPE가 이미 적용됨
                - self.U[layer_idx], self.SV[layer_idx] : low rank로 저장된 전체 key state들 
                - self.k_landmark[layer_idx]            : chunk 대표 vector들
                - self.k_landmark_idx[layer_idx]        : chunk index들
                
                prefill 후 CPU Mem. 에 저장되는 것들 :
                - self.v_cache_cpu[layer_idx]            : prefill 전체 value state들 저장
                """
                if self.minference == True:
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

            else: # decode
                # rope query and key
                query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                """
                query_states.shape:
                    (bsz, num_heads, 1, head_dim)
                key_states.shape:
                    (bsz, num_key_value_heads, 1, head_dim)
                """

                # update kv cache to buffer
                self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
                """
                self.{k/v}_cache_buffer[layer_idx]에 decoding stage에 생성된 새로운 state이 직접적으로 추가
                * 참고 : decoding stage의 새로운 key와 value state가 U[layer_idx], v_cache_cpu[layer_idx]에 추가되는 것이 아님
                * 참고 : 매 decoding step의 key state를 low rank로 저장하고 싶으면 self.SV이랑 곱한 다음 self.U에 추가
                """

                # get retrieval idx
                position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)
                """
                같은 kv group의 attention head들이 같은 sparse_retrieved v state들을 가지게 됨
                position_ids.shape:
                    (bsz, num_key_value_heads, sparse_budget)
                """

                # multi-stream
                curr_stream = torch.cuda.current_stream()
                get_value_stream = self.kv_cache.copy_stream

                # 1. get_value_stream이 curr_stream을 기다림
                with torch.cuda.stream(get_value_stream):
                    get_value_stream.wait_stream(curr_stream) # curr_stream 이전 작업 완료 대기
                    value_states = self.kv_cache.get_value_cache(layer_idx, position_ids) # V 복사 시작
                    """
                    v_cache_cpu에서 v_cache_buffer의 sparse_retrieved 구간으로 이동시킨 다음
                    v_cache_buffer 속 state들 [prefill_local | outlier_chunks | sparse_retrieved | generated_tokens] 모두 반환
                    """
                
                # 2. curr_stream은 즉시 K 계산 진행 (병렬)
                key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)
                """
                U에서 필요한 key state들만 고름 -> 차원 확장을 위하 SV와 곱함 -> RoPE 적용 -> k_cache_buffer의 sparse_retrieved 구간으로 이동시킨 다음
                k_cache_buffer 속 state들 [prefill_local | outlier_chunks | sparse_retrieved | generated_tokens] 모두 반환
                """

                # 3. curr_stream이 get_value_stream 완료 대기
                curr_stream.wait_stream(get_value_stream)

                # flash attention
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
        
        elif isinstance(self.kv_cache, SpeCacheKVCache):
            # curr_stream = None
            # prefetch_stream = None

            # if layer_idx == 0:
            #     breakpoint()

            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

            if q_len > 2: # prefill stage    
                """
                query_states.shape:
                    (bsz, num_heads, seq_len, head_dim)
                key_states.shape:
                    (bsz, num_key_value_heads, seq_len, head_dim)
                """
                
                self.kv_cache.prefill_kv_cache(key_states, value_states, layer_idx)

                if self.minference == True:
                    hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                else:
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
                    """
                    hidden_states.shape:
                        (bsz, seq_len, num_heads, head_dim)
                    """

            elif q_len == 1: # pre-decoding stage
                """
                query_states.shape:
                    (bsz, num_heads, 1, head_dim)
                key_states.shape:
                    (bsz, num_key_value_heads, 1, head_dim)
                """

                # Algorithm 2 : K = [K', K_1], V = [V', V_1]
                k_prime = self.kv_cache.k_cache_gpu.dequant_cache(layer_idx)
                v_prime = self.kv_cache.v_cache_gpu.dequant_cache(layer_idx)

                assert k_prime.shape == v_prime.shape, f"[Pre-decoding] : Shape mismatch between dequantized k and v cache, {k_prime.shape} vs {v_prime.shape}"

                key_states = torch.cat([k_prime, key_states], dim=2)
                value_states = torch.cat([v_prime, value_states], dim=2)
                
                # Algorithm 2 : A = Softmax(Q_1K^T)
                # -> flash_attn_func(return_attn_probs=True) : This option is for testing only. The returned probabilities are not guaranteed to be correct
                if self.num_heads > self.num_key_value_heads:
                    key_states = repeat_kv(key_states, self.num_key_value_groups)

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (self.head_dim ** -0.5)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                """
                attn_weights.shape:
                    (bsz, num_heads, 1, seq_len)
                """

                # Algorithm 2 : κ_1 = ArgTopK(A)
                # -> For GQA, we take the average of the Attention scores across all Attention Heads within each group (i.e., corresponding to one Key/Value pair) before applying ArgTopK.
                attn_weights_avg = attn_weights.view(bsz, self.num_key_value_heads, self.num_key_value_groups, 1, -1).mean(dim=2)
                attn_weights_avg = attn_weights_avg[:,:,:, :-1]
                """
                attn_weights_avg.shape:
                    (bsz, num_key_value_heads, 1, seq_len-1)
                """

                if self.kv_cache.arg_top_k > attn_weights_avg.size(-1):
                    raise ValueError(f"Pre-decoding Error : top_k {self.kv_cache.arg_top_k} > current kv length {attn_weights_avg.size(-1)}")
                
                _, retrieval_idx = torch.topk(attn_weights_avg, k=self.kv_cache.arg_top_k, dim=-1) # (bsz, num_key_value_heads, 1, top_k)
                """
                retrieval_idx.shape:
                    (bsz, num_key_value_heads, 1, top_k)
                """
                
                # Algorithm 2 : Pre-fetch(C_{κ_1})
                # (4.4. Efficiency) Note that our CPU-GPU interaction code is implemented using pytorch’s multi-stream mechanism and the Tensor.copy_() method, so the parallelism achieved is not theoretically optimal
                # curr_stream = torch.cuda.current_stream()
                # prefetch_stream = self.kv_cache.copy_stream

                # with torch.cuda.stream(prefetch_stream):
                #     prefetch_stream.wait_stream(curr_stream)
                #     self.kv_cache.prefetch_kv_beta(layer_idx, retrieval_idx)
                self.kv_cache.prefetch_kv_beta(layer_idx, retrieval_idx)

                # Algorithm 2 : O = AV (W_0 will be applied in post_attention_compute)
                if self.num_heads > self.num_key_value_heads:
                    value_states = repeat_kv(value_states, self.num_key_value_groups)

                hidden_states = torch.matmul(attn_weights, value_states).transpose(1, 2)
                """
                hidden_states.shape:
                    (bsz, 1, num_heads, head_dim)
                """


            elif q_len == 2: # decoding stage
                """
                query_states.shape:
                    (bsz, num_heads, 2, head_dim)
                key_states.shape:
                    (bsz, num_key_value_heads, 2, head_dim)
                """
                
                K_t_0 = key_states[:,:, 0:1]   # (bsz, num_key_value_heads, 1, head_dim)
                V_t_0 = value_states[:,:, 0:1] # (bsz, num_key_value_heads, 1, head_dim)

                # Speculative Token Top-K KV Cache Hit Rate: 
                #   The proportion of the top-k KV cache needed for the next output token that is hit by the top-k KV cache of the speculative token.
                if self.check_hit_rate:

                    current_query = query_states[:,:, 0:1] # (bsz, num_heads, 1, head_dim)
                    full_precision_keys = self.kv_cache.k_cache_cpu[layer_idx][:, :, :self.kv_cache.kv_offset] # (bsz, num_key_value_heads, seq_len, head_dim)
                    
                    if self.num_heads > self.num_key_value_heads:
                        full_precision_keys = repeat_kv(full_precision_keys, self.num_key_value_groups) 
                    
                    attn_weights_gt = torch.matmul(current_query, full_precision_keys.transpose(2,3)) * (self.head_dim ** -0.5) # (bsz, num_heads, 1, seq_len)
                    
                    _, top_k_gt = torch.topk(attn_weights_gt, k=self.kv_cache.arg_top_k, dim=-1) # (bsz, num_heads, 1, arg_top_k)
                    top_k_gt = top_k_gt.squeeze(2).squeeze(0) # (num_heads, arg_top_k)

                    top_k_spec = self.kv_cache.retrieve_buffer_idx[layer_idx] # (num_key_value_heads, arg_top_k)
                    top_k_spec = top_k_spec.repeat_interleave(self.num_key_value_groups, dim=0) # (num_heads, arg_top_k)

                    matches = (top_k_spec.unsqueeze(2) == top_k_gt.unsqueeze(1)) # (num_heads, arg_top_k, arg_top_k)
                    intersection_counts = matches.any(dim=2).sum(dim=1, keepdim=True).to("cpu") # (num_heads, 1)

                    assert (intersection_counts <= self.kv_cache.arg_top_k).all(), f"Hit count {intersection_counts.max().item()} exceeds top_k {self.kv_cache.arg_top_k}"
                    
                    global top_k_hitrates
                    top_k_hitrates[layer_idx] = torch.cat([top_k_hitrates[layer_idx], intersection_counts], dim=1)


                # Algortithm 3 : K = [K' U K_{κ_t}, K_t],  V = [V' U V_{κ_t}, V_t]
                keys_p, values_p = self.kv_cache.get_mixed_precision_kv_beta(layer_idx)

                assert keys_p.shape == values_p.shape, f"[Decoding stage] : Shape mismatch between mixed-precision k and v cache, {keys_p.shape} vs {values_p.shape}"
            
                key_states = torch.cat([keys_p, key_states], dim=2)
                value_states = torch.cat([values_p, value_states], dim=2)
                
                # Algorithm 3 : A = MaskedSoftmax(Q_tK^T)
                if self.num_heads > self.num_key_value_heads:
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (self.head_dim ** -0.5)
                attn_weights[:,:,0,-1] = float('-inf') # causal mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Algorithm 3 : κ_{t+1} = ArgTopK(A[1,:])
                # -> For GQA, we take the average of the Attention scores across all Attention Heads within each group (i.e., corresponding to one Key/Value pair) before applying ArgTopK.
                attn_weights_avg = attn_weights[:,:,-1].view(bsz, self.num_key_value_heads, self.num_key_value_groups, 1, -1).mean(dim=2)
                attn_weights_avg = attn_weights_avg[:,:,:,:-1] # remove the last token (which is the current speculative token)
                """
                attn_weights_avg.shape:
                    (bsz, num_key_value_heads, 1, seq_len-1)
                """
                if self.kv_cache.arg_top_k > attn_weights_avg.size(-1):
                    raise ValueError(f"Pre-decoding Error : top_k {self.kv_cache.arg_top_k} > current kv length {attn_weights_avg.size(-1)}")
                
                _, retrieval_idx = torch.topk(attn_weights_avg, k=self.kv_cache.arg_top_k, dim=-1) # (bsz, num_key_value_heads, 1, top_k)
                """
                retrieval_idx.shape:
                    (bsz, num_key_value_heads, 1, top_k)
                """
                                
                # Algorithm 3 : K'_t = Quant(K_t), V'_t = Quant(V_t)
                #               C_t = {V_t, K_t}, C'_t = {V'_t, K'_t}
                #               C = [C, offload(C_t)], C' = [C', C'_t]
                self.kv_cache.update_kv_cache_cpu(K_t_0, V_t_0, layer_idx)
                self.kv_cache.update_kv_cache_gpu(K_t_0, V_t_0, layer_idx)

                # Algorithm 3 : Pre-fetch(C_{κ_{t+1}})
                # (4.4. Efficiency) Note that our CPU-GPU interaction code is implemented using pytorch’s multi-stream mechanism and the Tensor.copy_() method, so the parallelism achieved is not theoretically optimal
                # curr_stream = torch.cuda.current_stream()
                # prefetch_stream = self.kv_cache.copy_stream

                # with torch.cuda.stream(prefetch_stream):
                #     prefetch_stream.wait_stream(curr_stream)
                #     self.kv_cache.prefetch_kv_beta(layer_idx, retrieval_idx)

                self.kv_cache.prefetch_kv_beta(layer_idx, retrieval_idx)

                # Algorithm 3 : O_t = AV (W_0 will be applied in post_attention_compute)
                if self.num_heads > self.num_key_value_heads:
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                
                hidden_states = torch.matmul(attn_weights, value_states).transpose(1, 2)
                """
                hidden_states.shape:
                    (bsz, 2, num_heads, head_dim)
                """

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            """
             bsz × seq_len가 64K를 초과할 때 전체 tensor를 한 번에 처리하는 대신 작은 chunk로 나누어 처리함
            """
            output = torch.empty_like(hidden_states)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)
            
            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        # if curr_stream is not None and prefetch_stream is not None:
        #     curr_stream.wait_stream(prefetch_stream)
        #     prefetch_stream = None
        #     curr_stream = None

        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False, hit_rate_path= None):
        """accuracy eval usage, not for throughput eval"""
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        if self.check_hit_rate:
            global top_k_hitrates
            top_k_hitrates = dict()
            for l in range(self.config.num_hidden_layers):
                top_k_hitrates[l] = torch.zeros(0)

        # prefill
        if cont == False:
            if input_ids.size(1) >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            # if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
            #     raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            # logits = self.prefill_cont(input_ids)
            raise NotImplementedError("LLM.generate() : cont=True is not supported for now")
        
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())

        self.kv_cache.H2D()
        
        # pre-decoding
        if isinstance(self.kv_cache, SpeCacheKVCache):
            logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
            speculative_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            # Speculative Token Exact Hit Rate: 
            #   The proportion of speculative tokens that perfectly match the next output token.
            if self.check_hit_rate:
                spec_token_ids = []
                spec_token_ids.extend(speculative_token[0].tolist())

        if benchmark == True:
            start = time.time()
        
        while n < gen_len:
            if input_ids.size(1) + n >= self.max_length:
                print(f"Reached max length {self.max_length}, stopping generation.")
                break

            if isinstance(self.kv_cache, SpeCacheKVCache):
                decode_input = torch.cat([next_token, speculative_token], dim=-1)
                logits = self.inference(input_ids=decode_input, position_ids=self.get_ctx(decode_input))
                next_token = sample_token(logits[:, 0, :], temperature=temperature, top_p=top_p, top_k=top_k)
                speculative_token = sample_token(logits[:, 1, :], temperature=temperature, top_p=top_p, top_k=top_k)
     
            else :
                logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
                next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            
            if self.check_hit_rate and isinstance(self.kv_cache, SpeCacheKVCache):
                spec_token_ids.extend(speculative_token[0].tolist())

            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        if not isinstance(self.kv_cache, SpeCacheKVCache):
            # feed new token to the model
            self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.check_hit_rate and (hit_rate_path is not None):

            hit_rate_result = {
                "decoded_output": [self.tokenizer.decode(generated_ids, skip_special_tokens=True)],
                "generated_ids": generated_ids,
                "spec_token_ids": spec_token_ids,
                "top_k_hitrates": top_k_hitrates
            }

            torch.save(hit_rate_result, hit_rate_path)
 
        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    

    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        pass

    @torch.inference_mode()
    def warmup(self):
        pass

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False):
        pass