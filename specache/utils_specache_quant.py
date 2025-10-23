import torch
import torch.nn.functional as F
import numpy as np

"""
process_input_by_channel(...):
    args 
        input.shape :
            (bsz, seq_len, dim = num_head*head_dim)

    return
        input_groups.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, group_size)
        {mn/mx}.shape:
            (num_head*head_dim, (bsz*seq_len)//group_size)
"""
def process_input_by_channel(input, group_size, do_padding=False):
    num_features = input.shape[-1]
  
    input_flatten = input.view(-1, num_features).transpose(0, 1) # (bsz, seq_len, dim) -> (dim, bsz*seq_len)
    num_instances = input_flatten.shape[-1]
    # Compute min, max by groups
    if num_instances % group_size != 0:
        if do_padding:
            # Padding
            new_num_instances = (num_instances // group_size + 1) * group_size
            delta = new_num_instances - num_instances
            input_flatten = torch.cat([input_flatten,
                                    torch.zeros([num_features, delta], dtype=input.dtype, device=input.device)], 1)
        else:
            raise Exception("seq_len*bsz should be divisible by group_size")
        
    input_groups = input_flatten.reshape(-1, group_size) # (dim, bsz*seq_len) -> (dim*(bsz*seq_len)//group_size, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(num_features, -1, group_size), mn.view(num_features, -1), mx.view(num_features, -1)

"""
process_input_by_token(...):
    args 
        input.shape :
            (bsz, seq_len, dim = num_head*head_dim)

    return
        input_groups.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, group_size)
        {mn/mx}.shape:
            (bsz*seq_len, (num_head*head_dim)//group_size)
"""
def process_input_by_token(input, group_size):
    num_features = input.shape[-1] # num_head * head_dim
  
    input_flatten = input.view(-1, num_features) # (bsz, seq_len, dim) -> (bsz*seq_len, dim)
    num_instances = input_flatten.shape[0] # bsz * seq_len
    # Compute min, max by groups
    if num_features % group_size != 0:
        raise Exception("num_head*head_dim should be divisible by group_size")
        
    input_groups = input_flatten.reshape(-1, group_size) # (bsz*seq_len, dim) -> ((bsz*seq_len)*dim//group_size, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(num_instances, -1, group_size), mn.view(num_instances, -1), mx.view(num_instances, -1)

"""
quantize_and_pack_cache(...):
    args
        input.shape :
            (bsz, num_head, seq_len, head_dim)

    return (method='channel')
        quantized.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, group_size)
        scale.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, 1)
        zp.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, 1)

    return (method='token')
        quantized.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, group_size)
        scale.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, 1)
        zp.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, 1)
"""
def quantize_and_pack_cache(input, method, group_size, num_bits, simulate=False, padding=False, do_specache_ver=False):    
    assert len(input.shape) == 4
    bsz, _, seq_len, _ = input.shape
    input = input.transpose(1, 2).reshape(bsz, seq_len, -1) # (bsz, num_head, seq_len, head_dim) -> (bsz, seq_len, num_head*dim_head)
    
    if method == 'channel':
        input_groups, mn, mx = process_input_by_channel(input, group_size, padding)
        """
        input_groups.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, group_size)
        {mn/mx}.shape:
            (num_head*head_dim, (bsz*seq_len)//group_size)
        """
    elif method == 'token':
        input_groups, mn, mx = process_input_by_token(input, group_size)
        """
        input_groups.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, group_size)
        {mn/mx}.shape:
            (bsz*seq_len, (num_head*head_dim)//group_size)
        """
    else:
        raise Exception("Unsupported quantization method")
    
    if simulate:
        mn, mx = mn.unsqueeze(-1), mx.unsqueeze(-1)

        if do_specache_ver:
            if num_bits != 1:
                raise Warning("SpeCache quantization only supports 1-bit quantization")
            
            # print(f"Using SpeCache {num_bits}-bit per-{method} quantization")
            zp = (3*mn + mx) / 4
            scale = (mx - mn) / 2
        else:
            # print(f"Using KIVI {num_bits}-bit per-{method} quantization")
            zp = mn
            scale = (mx - mn) / (2 ** num_bits - 1)

        input_groups = (input_groups - zp) / scale
        input_groups = F.relu(input_groups) # for clipping negative values to zero
        rounded_input = input_groups.round_()
        return rounded_input.to(torch.int8), scale, zp
    else:
        raise NotImplementedError

"""
dequantize_and_unpack_cache(...):
    args (if method='channel')
        data.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, group_size)
        scale.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, 1)
        zp.shape: 
            (num_head*head_dim, (bsz*seq_len)//group_size, 1)

    args (if method='token')
        data.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, group_size)
        scale.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, 1)
        zp.shape: 
            (bsz*seq_len, (num_head*head_dim)//group_size, 1)
            
    return
        data.shape :
            (bsz, num_head, seq_len, head_dim)
"""
def dequantize_and_unpack_cache(data, scale, zp, method, bsz, num_head, head_dim, simulate=False):
    assert len(data.shape) == 3

    if method == 'channel':
        seq_len = (data.shape[1] * data.shape[2]) // bsz
    elif method == 'token':
        seq_len = data.shape[0] // bsz
    
    shape = (bsz , num_head, seq_len, head_dim)
    num_feats = shape[1] * shape[3]         # num_head * head_dim
    ori_num_instances = shape[0] * shape[2] # bsz * seq_len
  
    if simulate:
        data = data * scale + zp  # (num_head*head_dim, (bsz*seq_len)//group_size, group_size)
    else:
        raise NotImplementedError

    if method == 'channel':
        dequantized_input = data.view(num_feats, -1) # (num_head*head_dim, bsz*seq_len)
        
        if ori_num_instances != dequantized_input.shape[1]:
            dequantized_input = dequantized_input[:, 0:ori_num_instances]
        
        data = dequantized_input.transpose(0, 1).view(shape[0], -1, num_feats) # (bsz, seq_len, num_head*head_dim)
        data = data.view(shape[0], shape[2], shape[1], -1).transpose(1, 2)     # (bsz, num_head, seq_len, head_dim)
    elif method == 'token':
        dequantized_input = data.view(ori_num_instances, -1) # (bsz*seq_len, num_head*head_dim)
 
        data = dequantized_input.view(shape[0], -1, num_feats)                 # (bsz, seq_len, num_head*head_dim)
        data = data.view(shape[0], shape[2], shape[1], -1).transpose(1, 2)     # (bsz, num_head, seq_len, head_dim)
    else:
        raise Exception("Unsupported dequantization method")
    
    assert data.shape == shape

    return data

"""
concat_quant(...): 
        args (if method='channel')
            quant{1/2}.shape:
                (num_head*head_dim, (bsz*seq_len{1/2})//group_size, group_size)
            scale{1/2}.shape:
                (num_head*head_dim, (bsz*seq_len{1/2})//group_size, 1)
            zp{1/2}.shape:
                (num_head*head_dim, (bsz*seq_len{1/2})//group_size, 1)
        return
            quant_new.shape:
                (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size, group_size)
            scale_new.shape:
                (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size, 1)
            zp_new.shape:
                (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size, 1)

        args (if method='token')
            quant{1/2}.shape:
                (bsz*seq_len{1/2}, (num_head*head_dim)//group_size, group_size)
            scale{1/2}.shape:
                (bsz*seq_len{1/2}, (num_head*head_dim)//group_size, 1)
            zp{1/2}.shape:
                (bsz*seq_len{1/2}, (num_head*head_dim)//group_size, 1)
        return
            quant_new.shape:
                (bsz*(seq_len1+seq_len2), (num_head*head_dim)//group_size, group_size)
            scale_new.shape:
                (bsz*(seq_len1+seq_len2), (num_head*head_dim)//group_size, 1)
            zp_new.shape:
                (bsz*(seq_len1+seq_len2), (num_head*head_dim)//group_size, 1)
"""
def concat_quant(quant1, scale1, zp1, quant2, scale2, zp2, bsz, num_head, head_dim, method):
    
    if len(quant1.shape) != 3:
        return quant2, scale2, zp2

    if len(quant2.shape) != 3 :
        return quant1, scale1, zp1

    num_feats = num_head * head_dim

    if quant1.shape[-1] == quant2.shape[-1]:
        group_size = quant1.shape[-1]
    else:
        raise Exception("group_size should be the same for two quantized tensors")
    
    if method == "channel":
        quant1 = quant1.view(num_feats, -1)                     # (num_head*head_dim, bsz*seq_len)
        quant1 = quant1.transpose(0,1).view(bsz, -1, num_feats) # (bsz, seq_len, num_head*head_dim)
        quant2 = quant2.view(num_feats, -1) 
        quant2 = quant2.transpose(0,1).view(bsz, -1, num_feats)

        quant_new = torch.cat([quant1, quant2], 1)               # (bsz, seq_len1+seq_len2, num_head*head_dim)
        quant_new = quant_new.view(-1, num_feats).transpose(0,1) # (num_head*head_dim, bsz*(seq_len1+seq_len2))
        quant_new = quant_new.view(num_feats, -1, group_size)    # (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size, group_size)

        scale1 = scale1.view(num_feats, -1)                     # (num_head*head_dim, (bsz*seq_len1)//group_size)
        scale1 = scale1.transpose(0,1).view(bsz, -1, num_feats) # (bsz, (seq_len1)//group_size, num_head*head_dim)
        scale2 = scale2.view(num_feats, -1)
        scale2 = scale2.transpose(0,1).view(bsz, -1, num_feats)
        
        scale_new = torch.cat([scale1, scale2], 1)               # (bsz, (seq_len1+seq_len2)//group_size, num_head*head_dim)
        scale_new = scale_new.view(-1, num_feats).transpose(0,1) # (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size)
        scale_new = scale_new.view(num_feats, -1, 1)             # (num_head*head_dim, bsz*(seq_len1+seq_len2)//group_size, 1)  

        zp1 = zp1.view(num_feats, -1)                     
        zp1 = zp1.transpose(0,1).view(bsz, -1, num_feats)  
        zp2 = zp2.view(num_feats, -1)  
        zp2 = zp2.transpose(0,1).view(bsz, -1, num_feats) 
        
        zp_new = torch.cat([zp1, zp2], 1)                  
        zp_new = zp_new.view(-1, num_feats).transpose(0,1)
        zp_new = zp_new.view(num_feats, -1, 1)

    elif method == "token":
        quant1 = quant1.view(-1, num_feats)                     # (bsz*seq_len1, num_head*head_dim)
        quant1 = quant1.view(bsz, -1, num_feats)                # (bsz, seq_len1, num_head*head_dim)
        quant2 = quant2.view(-1, num_feats) 
        quant2 = quant2.view(bsz, -1, num_feats)

        quant_new = torch.cat([quant1, quant2], 1)                     # (bsz, seq_len1+seq_len2, num_head*head_dim)
        quant_new = quant_new.view(-1, num_feats)                      # (bsz*(seq_len1+seq_len2), num_head*head_dim)
        quant_new = quant_new.view(quant_new.shape[0], -1, group_size) # (bsz*(seq_len1+seq_len2), (num_head*head_dim)//group_size, group_size)

        scale1 = scale1.squeeze(-1)                              # (bsz*seq_len1, num_head*head_dim//group_size)
        scale1 = scale1.view(bsz, -1, num_feats//group_size)     # (bsz, seq_len1, num_head*head_dim//group_size)
        scale2 = scale2.squeeze(-1)
        scale2 = scale2.view(bsz, -1, num_feats//group_size)

        scale_new = torch.cat([scale1, scale2], 1)               # (bsz, (seq_len1+seq_len2), num_head*head_dim//group_size)
        scale_new = scale_new.view(-1, num_feats//group_size)    # (bsz*(seq_len1+seq_len2), num_head*head_dim//group_size)
        scale_new = scale_new.unsqueeze(-1)                      # (bsz*(seq_len1+seq_len2), num_head*head_dim//group_size, 1)

        zp1 = zp1.squeeze(-1)
        zp1 = zp1.view(bsz, -1, num_feats//group_size)
        zp2 = zp2.squeeze(-1)
        zp2 = zp2.view(bsz, -1, num_feats//group_size)
        
        zp_new = torch.cat([zp1, zp2], 1)
        zp_new = zp_new.view(-1, num_feats//group_size)
        zp_new = zp_new.unsqueeze(-1)
    else:
        raise Exception("Unsupported quantization method")             

    return quant_new.to(torch.int8), scale_new, zp_new

def make_synthetic_key_cache(bsz, num_head, seq_len, head_dim, outlier_indices, device='cuda'):
    input1 = torch.randn((bsz, num_head, seq_len, head_dim), dtype=torch.float16, device='cuda')
    input1 = torch.clamp(input1, min=-1, max=1)

    outlier_values = torch.rand((bsz, num_head, seq_len, len(outlier_indices)), dtype=torch.float16, device='cuda') * 50 + 300  # 300~350 범위

    input1[:, :, :, outlier_indices] = outlier_values

    return input1

def test_quantize():
    group_size = 32
    bit_width = 2
    do_specache=False

    # input = torch.randn((2, 4, group_size*2, 128), dtype=torch.float16, device='cuda')

    outlier_indices = np.random.choice(128, size=128//10 , replace=False)
    input = make_synthetic_key_cache(2, 4, group_size*2, 128, outlier_indices, device='cuda')
    
    # ===========================================================================================
    # Per-token quantization & dequantization
    quantized_t, scale_t, zp_t = quantize_and_pack_cache(input, 
                                                         "token", 
                                                         group_size, 
                                                         bit_width, 
                                                         simulate=True, 
                                                         padding=False, 
                                                         do_specache_ver=do_specache)
    
    dequantized_t = dequantize_and_unpack_cache(quantized_t, 
                                                scale_t, 
                                                zp_t, 
                                                "token", 
                                                2, 
                                                4, 
                                                128,
                                                simulate=True)
    
    print("per-token quant error : ", torch.mean((input - dequantized_t)**2).item())

    # ===========================================================================================
    # Per-channel quantization & dequantization
    quantized_c, scale_c, zp_c = quantize_and_pack_cache(input, 
                                                         "channel",
                                                         group_size, 
                                                         bit_width, 
                                                         simulate=True, 
                                                         padding=False, 
                                                         do_specache_ver=do_specache)
    
    dequantized_c = dequantize_and_unpack_cache(quantized_c, 
                                                scale_c, 
                                                zp_c, 
                                                "channel",
                                                2, 
                                                4, 
                                                128,
                                                simulate=True)
    
    print("per-chnl quant error : ", torch.mean((input - dequantized_c)**2).item())

def test_concat_quant():
    bsz = 2
    num_head = 4
    head_dim = 128
    group_size = 32
    bit_width = 4
    do_specache=False
    
    # input1 = torch.randn((bsz, num_head, group_size*2, head_dim), dtype=torch.float16, device='cuda')
    # input2 = torch.randn((bsz, num_head, 1, head_dim), dtype=torch.float16, device='cuda')

    outlier_indices = np.random.choice(head_dim, size=head_dim//10 , replace=False)
    input1 = make_synthetic_key_cache(bsz, num_head, group_size*2, head_dim, outlier_indices, device='cuda')
    input2 = make_synthetic_key_cache(bsz, num_head, group_size, head_dim, outlier_indices, device='cuda')

    gt_input = torch.cat([input1, input2], 2)
    # ===========================================================================================
    # Per-token quantization 
    quantized_v1, scale1, zp1 = quantize_and_pack_cache(input1, "token", group_size, bit_width, simulate=True, padding=False, do_specache_ver=do_specache)
    quantized_v2, scale2, zp2 = quantize_and_pack_cache(input2, "token", group_size, bit_width, simulate=True, padding=False, do_specache_ver=do_specache)
    
    # Per-token concat 
    quant_new, scale_new, zp_new = concat_quant(quantized_v1, scale1, zp1, 
                                                quantized_v2, scale2, zp2, 
                                                bsz, num_head, head_dim, method="token")
    
    # Per-token dequantization
    dequantized_v = dequantize_and_unpack_cache(quant_new, scale_new, zp_new, "token", bsz, num_head, head_dim, simulate=True)
    print(">> Concat per-token quant error : ", torch.mean((gt_input - dequantized_v)**2).item())

    # ===========================================================================================
    # Per-channel quantization 
    quantized_v1, scale1, zp1 = quantize_and_pack_cache(input1, "channel", group_size, bit_width, simulate=True, padding=False, do_specache_ver=do_specache)
    quantized_v2, scale2, zp2 = quantize_and_pack_cache(input2, "channel", group_size, bit_width, simulate=True, padding=False, do_specache_ver=do_specache)
    
    # Per-channel concat 
    quant_new, scale_new, zp_new = concat_quant(quantized_v1, scale1, zp1, 
                                                quantized_v2, scale2, zp2, 
                                                bsz, num_head, head_dim, method="channel")
    
    # Per-channel dequantization
    dequantized_v = dequantize_and_unpack_cache(quant_new, scale_new, zp_new, "channel", bsz, num_head, head_dim, simulate=True)
    print(">> Concat per-chnl quant error : ", torch.mean((gt_input - dequantized_v)**2).item())
    

class PerChannelQuantCache:
    def __init__(self, 
                 quant_group_size,
                 quant_bit,
                 do_specache_quant,
                 num_key_value_heads, 
                 batch_size, 
                 quant_residual, 
                 head_dim, 
                 num_layers, 
                 device, 
                 dtype):
        
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.batch_size = batch_size
        self.quant_residual = quant_residual
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype


        self.quant_group_size = quant_group_size
        self.quant_bit = quant_bit
        self.do_specache_quant = do_specache_quant


        self.quantized = [torch.zeros(0, device=self.device, dtype=torch.int8) for _ in range(self.num_layers)]
        self.zp = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.scale = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        
        self.residual = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.num_key_value_heads,
            self.quant_residual-1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        )

        self.kv_offset = 0
        self.prefill_cnt = 0
        self.enqueue_ptr = 0
        self.quant_bound_idx = 0
    
    def prefill(self, layer_idx, new_states):
        """
        new_states.shape:
            (bsz, num_key_value_heads, seq_len, head_dim)
        """
        assert len(new_states.shape) == 4
        incoming = new_states.shape[-2]

        residual_count = incoming % self.quant_residual # 0 ~ (quant_residual-1)
        quant_bound_idx = incoming - residual_count

        if quant_bound_idx == 0:
            # if incoming < quant_residual, no quantization, all goes into residuals
            self.residual[layer_idx][:, :, :incoming] = new_states.clone()
           
            if layer_idx == self.num_layers - 1:
                self.enqueue_ptr = incoming
        
        else:
            self.residual[layer_idx][:, :, :residual_count] = new_states[:, :, quant_bound_idx:, :].clone()
            
            if layer_idx == self.num_layers - 1:
                self.enqueue_ptr = residual_count

            states_quant = new_states[:, :, :quant_bound_idx, :]

            self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx] = quantize_and_pack_cache(states_quant, 
                                                                                                            "channel", 
                                                                                                            self.quant_group_size, 
                                                                                                            self.quant_bit, 
                                                                                                            simulate=True, 
                                                                                                            padding=False, 
                                                                                                            do_specache_ver=self.do_specache_quant)
            
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.prefill_cnt = incoming
            self.quant_bound_idx = quant_bound_idx

    def dequant_cache(self, layer_idx):

        if self.quant_bound_idx == 0 :
            return self.residual[layer_idx]
        
        deq = dequantize_and_unpack_cache(self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx], "channel",
                                                self.batch_size, self.num_key_value_heads, self.head_dim, simulate=True)

        if self.enqueue_ptr == 0:
            return deq
        
        res = self.residual[layer_idx][:, :, :self.enqueue_ptr]
        
        return torch.cat([deq, res], dim=-2)

    def update(self, layer_idx, new_state):
        """
        new_state.shape:
            (bsz, num_key_value_heads, 1, head_dim)
        """
        incoming = new_state.shape[-2]

        if self.enqueue_ptr != self.quant_residual - 1:
            self.residual[layer_idx][:, :, self.enqueue_ptr:self.enqueue_ptr+incoming].copy_(new_state)
            
            if layer_idx == self.num_layers - 1:
                self.enqueue_ptr += incoming
        
        else: # if residual is full
            full_residual = torch.concat([self.residual[layer_idx], new_state], dim=-2)

            self.residual[layer_idx].zero_()

            if layer_idx == self.num_layers - 1:
                self.enqueue_ptr = 0
                self.quant_bound_idx += self.quant_residual
            
            q, s, z = quantize_and_pack_cache(full_residual, 
                                                    "channel", 
                                                    self.quant_group_size, 
                                                    self.quant_bit, 
                                                    simulate=True, 
                                                    padding=False, 
                                                    do_specache_ver=self.do_specache_quant)
            
            self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx] = concat_quant(self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx], 
                                                                                                     q, s, z,
                                                                                                     self.batch_size, self.num_key_value_heads, self.head_dim, "channel")

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming

    def clear(self):
        self.quantized = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.zp = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.scale = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.residual.zero_()

        self.kv_offset = 0
        self.prefill_cnt = 0
        self.enqueue_ptr = 0
        self.quant_bound_idx  = 0

    def get_kv_len(self):
        return self.kv_offset

class PerTokenQuantCache:
    def __init__(self, 
                 quant_group_size,
                 quant_bit,
                 do_specache_quant,
                 num_key_value_heads, 
                 batch_size, 
                 quant_residual, 
                 head_dim, 
                 num_layers, 
                 device, 
                 dtype):
        
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.batch_size = batch_size
        self.quant_residual = quant_residual
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype


        self.quant_group_size = quant_group_size
        self.quant_bit = quant_bit
        self.do_specache_quant = do_specache_quant


        self.quantized = [torch.zeros(0, device=self.device, dtype=torch.int8) for _ in range(self.num_layers)]
        self.zp = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.scale = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        
        self.residual_queue = CircularQueue(num_layers, 
                                            batch_size, 
                                            num_key_value_heads, 
                                            quant_residual, 
                                            head_dim, 
                                            device, 
                                            dtype)
        

        self.kv_offset = 0
        self.prefill_cnt = 0
        self.quant_bound_idx = 0
 
    def prefill(self, layer_idx, new_states):
        """
        new_states.shape:
            (bsz, num_key_value_heads, seq_len, head_dim)
        """
        assert len(new_states.shape) == 4
        incoming = new_states.shape[-2]

        quant_bound_idx = incoming - self.quant_residual

        if incoming <= self.quant_residual:
            # no quantization, all in residuals
            self.residual_queue.fill_queue(layer_idx, new_states)
        else:
            states_resi = new_states[:, :, quant_bound_idx:, :]
            self.residual_queue.fill_queue(layer_idx, states_resi)
            
            states_quant = new_states[:, :, :quant_bound_idx, :]
            q_k, s_k, z_k = quantize_and_pack_cache(states_quant, 
                                        "token", 
                                        self.quant_group_size, 
                                        self.quant_bit, 
                                        simulate=True, 
                                        padding=False, 
                                        do_specache_ver=self.do_specache_quant)
            
            self.quantized[layer_idx] = q_k
            self.scale[layer_idx] = s_k
            self.zp[layer_idx] = z_k

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.prefill_cnt = incoming
            self.quant_bound_idx = quant_bound_idx if quant_bound_idx > 0 else 0

    def dequant_cache(self, layer_idx):
        if self.quant_bound_idx == 0:
            return self.residual_queue.get_full_queue(layer_idx)
        
        deq = dequantize_and_unpack_cache(self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx], "token",
                                          self.batch_size, self.num_key_value_heads, self.head_dim, simulate=True)

        if self.residual_queue.current_size == 0:
            return deq
        
        res = self.residual_queue.get_full_queue(layer_idx)
        
        return torch.cat([deq, res], dim=-2)

    def update(self, layer_idx, new_state):
        """
        new_state.shape:
            (bsz, num_key_value_heads, 1, head_dim)
        """
        incoming = new_state.shape[-2]

        popped = self.residual_queue.add(layer_idx, new_state)

        if popped is not None:
            q, s, z = quantize_and_pack_cache(popped, 
                                              "token", 
                                               self.quant_group_size, 
                                               self.quant_bit, 
                                               simulate=True, 
                                               padding=False, 
                                               do_specache_ver=self.do_specache_quant)
            
            self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx] = concat_quant(self.quantized[layer_idx], self.scale[layer_idx], self.zp[layer_idx], 
                                                                                                q, s, z,
                                                                                                self.batch_size, self.num_key_value_heads, self.head_dim, "token")  
            
            if layer_idx == self.num_layers - 1:
                self.quant_bound_idx += 1

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming

    def clear(self):
        self.quantized = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.zp = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.scale = [torch.zeros(0, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.residual_queue.clear()

        self.kv_offset = 0
        self.prefill_cnt = 0
        self.quant_bound_idx  = 0

    def get_kv_len(self):
        return self.kv_offset

class CircularQueue:
    def __init__(self, num_layers, batch_size, num_heads, residual_len, head_dim, device, dtype):
        self.residual_len = residual_len
        self.num_layers = num_layers
        
        self.my_queue = torch.zeros(
            (num_layers, batch_size, num_heads, residual_len, head_dim),
            device=device,
            dtype=dtype
        )
        
        self.enqueue_pos = 0
        self.current_size = 0

    def fill_queue(self, layer_idx, new_states):
        """
        new_states.shape:
            (bsz, num_heads, residual_len, head_dim)
        """
        assert new_states.shape[-2] <= self.residual_len, "CircularQueue:fill_queue() -- new_states.shape[-2] should be less than or equal to residual_len"

        prefill_len = new_states.shape[-2]
        self.my_queue[layer_idx][:, :, :prefill_len] = new_states.clone()
        
        if layer_idx == self.num_layers - 1:
            self.enqueue_pos = prefill_len % self.residual_len # 0 ~ (residual_len-1)
            self.current_size = prefill_len

    def add(self, layer_idx, new_s):
        """
        new_s.shape:
            (bsz, num_heads, 1, head_dim)
        """
        popped = None

        if self.is_full():
            popped = self.my_queue[layer_idx][:, :, self.enqueue_pos:self.enqueue_pos+1, :].clone()

        self.my_queue[layer_idx][:, :, self.enqueue_pos:self.enqueue_pos+1] = new_s

        if layer_idx == self.num_layers - 1:
            self.enqueue_pos = (self.enqueue_pos + 1) % self.residual_len

            if self.current_size < self.residual_len:
                self.current_size += 1
        
        return popped

    def is_full(self):
        return self.current_size == self.residual_len

    def get_full_queue(self, layer_idx):
        if self.current_size < self.residual_len:
            return self.my_queue[layer_idx][:, :, :self.current_size, :]
        else:
            old_part = self.my_queue[layer_idx][:, :, self.enqueue_pos:, :]
            new_part = self.my_queue[layer_idx][:, :, :self.enqueue_pos, :]
            return torch.cat([old_part, new_part], dim=2)

    def clear(self):
        self.my_queue.zero_()
        self.enqueue_pos = 0
        self.current_size = 0

# python specache/utils_specache_quant.py
if __name__ == '__main__':
    torch.set_printoptions(linewidth=230, sci_mode=False, edgeitems=5)
    test_quantize()
    # test_concat_quant()

    