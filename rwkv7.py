"""
Based on: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo_rnn.py
"""

from typing import List
from tinygrad import Tensor, TinyJit
from tinygrad.nn import GroupNorm
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.nn.state import torch_load
from tokenizers import Tokenizer
import numpy as np

class RWKV_RNN:
    @TinyJit
    def __init__(self, config):
        self.n_embedding = config.get("n_embd")
        self.n_layer = config.get("n_layer")
        self.n_head = config.get("n_head")
        self.head_size = config.get("head_size")
        self.vocab_size = config.get("vocab_size")
        self.model = {}
        
        model = torch_load(config.get("model_name"))

        #print(f"Devices = {', '.join(map(str, Device.get_available_devices()))}")
        #Device.DEFAULT = "AMD"
        #print(f"Device.DEFAULT = {Device.DEFAULT}")
        
        for key in model.keys():
            if key.endswith('att.w0'):
                self.model[key] = model[key].to(Device.DEFAULT).squeeze()#.float()
            elif key.endswith('att.r_k'):
                self.model[key] = model[key].to(Device.DEFAULT).flatten()
            elif key.endswith('att.a0'):
                self.model['blocks.0.att.v0'] = model[key].to(Device.DEFAULT).flatten()
                self.model[key] = model[key].to(Device.DEFAULT).squeeze()
            elif key.endswith('att.a1'):
                self.model['blocks.0.att.v1'] = model[key].to(Device.DEFAULT).flatten()
                self.model[key] = model[key].to(Device.DEFAULT).squeeze()
            elif key.endswith('att.a2'):
                self.model['blocks.0.att.v2'] = model[key].to(Device.DEFAULT).flatten()
                self.model[key] = model[key].to(Device.DEFAULT).squeeze()
            else:
                self.model[key] = model[key].to(Device.DEFAULT).squeeze()

        self.model['emb.weight'] = self.model['emb.weight'].to(Device.DEFAULT).layernorm() * self.model['blocks.0.ln0.weight'].to(Device.DEFAULT) + self.model['blocks.0.ln0.bias'].to(Device.DEFAULT)

    @TinyJit
    def init_state(self):
        init_state = [None for _ in range(self.n_layer * 3)]
        for i in range(self.n_layer):
            init_state[i*3+0] = Tensor.zeros(self.n_embedding, dtype=dtypes.bfloat16).to(Device.DEFAULT)
            init_state[i*3+1] = Tensor.zeros((self.n_embedding // self.head_size, self.head_size, self.head_size), dtype=dtypes.bfloat16).to(Device.DEFAULT) #.float()
            init_state[i*3+2] = Tensor.zeros(self.n_embedding, dtype=dtypes.bfloat16).to(Device.DEFAULT)
        return init_state

    def forward(self, token: int, state: List[Tensor]):
        z = self.model
        #print("Embedding size:", z['emb.weight'].shape)
        #raise NotImplementedError("Implement the forward pass")
        x = z['emb.weight'][token]
        v_first = Tensor.zeros_like(x)
        
        for i in range(self.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'
            
            # Replace LayerNorm class with tensor.layernorm()
            xx = x.layernorm() * z[bbb+'ln1.weight'] + z[bbb+'ln1.bias']
            
            xx, state[i*3+0], state[i*3+1], v_first = self.time_mixing(
                i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], 
                z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias']
            )
            #print(f"xx.dtype = {xx.dtype}")
            x = x + xx
            
            # Replace second LayerNorm
            xx = x.layernorm() * z[bbb+'ln2.weight'] + z[bbb+'ln2.bias']
            
            xx, state[i*3+2] = self.channel_mixing(
                xx, state[i*3+2], 
                z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight']
            )
            
            x = x + xx
        
        # Replace final LayerNorm
        x = x.layernorm() * z['ln_out.weight'] + z['ln_out.bias']
        
        # Matrix multiplication for final output
        x = z['head.weight'] @ x
        
        return x, state

    def time_mixing(self, layer_id: int, H: int, N: int, x, x_prev, v_first, state,
                    x_r, x_w, x_k, x_v, x_a, x_g,
                    w0, w1, w2, a0, a1, a2, v0, v1, v2,
                    g1, g2, k_k, k_a, r_k,
                    kw, vw, rw, ow, ln_w, ln_b):
        # Time-shift mixing
        xx = x_prev - x
        #print(f"xx.dtype = {xx.dtype}")
        
        # Compute x terms in parallel
        xr = x + xx * x_r
        xw = x + xx * x_w
        xk = x + xx * x_k
        xv = x + xx * x_v
        xa = x + xx * x_a
        xg = x + xx * x_g
        
        # Linear transformations
        r = rw @ xr
        w = (xw @ w1).tanh() @ w2
        k = kw @ xk
        v = vw @ xv
        
        # Attention computation
        a = (xa @ a1 @ a2 + a0).sigmoid()
        g = (xg @ g1 @ g2).sigmoid()
        
        # Process k
        kk = k * k_k
        kk = kk.reshape(H, N)
        kk_norm = (kk * kk).sum(axis=1, keepdim=True).sqrt()
        kk = (kk / kk_norm).reshape(-1)
        
        k = k * (1 + (a-1) * k_a)
        
        # Handle v_first
        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * (xv @ v1 @ v2 + v0).sigmoid()
        
        # Process w
        w = w0 + w
        w = (-0.606531 * w.sigmoid()).exp()  # 0.606531 = exp(-0.5)
        
        # RWKV-7 kernel
        vk = v.reshape(H, N, 1) @ k.reshape(H, 1, N)
        ab = (-kk).reshape(H, N, 1) @ (kk*a).reshape(H, 1, N)
        
        # State update in single operation
        state = state * w.reshape(H, 1, N) + state @ ab + vk
        
       # Output computation with group norm
        out = state @ r.reshape(H, N, 1)
        out = out.reshape(1, H*N)
            
        # Apply group normalization - notice ln_w and ln_b are used here
        gn = GroupNorm(H, H*N, eps=64e-5)
        # Set the weights and bias
        gn.weight = ln_w
        gn.bias = ln_b
        out = gn(out)
        out = out.reshape(H*N)
        
        # Final computation matching PyTorch version exactly
        out = out + ((r * k * r_k).reshape(H, N).sum(axis=1, keepdim=True) * v.reshape(H, N)).reshape(H*N)
        
        # Final output
        return ow @ (out * g), x, state, v_first
    
    def channel_mixing(self, x, x_prev, x_k, kw, vw):
        #print(f"x.dtype = {x.dtype}, x_prev.dtype = {x_prev.dtype}, x_k.dtype = {x_k.dtype}, kw.dtype = {kw.dtype}, vw.dtype = {vw.dtype}")
        xx = x_prev - x
        k = x + xx * x_k
        k = (kw @ k).relu() ** 2
        return vw @ k, x

@TinyJit
def sample_logits(logits: Tensor):
    # Get the index of the maximum value
    top_token = logits.numpy()
    print("top_token: ", top_token)
    return np.argmax(top_token)

@TinyJit
def config_from_file(file: str):
    model = torch_load(file)

    vocab_size, n_embd = model['emb.weight'].shape
    n_layer = max(int(k.split('.')[1]) for k in model.keys() if k.startswith('blocks.'))
    n_head, head_size = model['blocks.0.att.r_k'].shape
    
    config = {
        'n_embd': n_embd,
        'n_layer': n_layer,
        'vocab_size': vocab_size,
        'head_size': head_size,
        'n_head': n_head,
        'model_name': file
    }
    return config

model = RWKV_RNN(config_from_file('RWKV-x070-Pile-164M-L33-D512-20241218-ctx4096.pth'))
init_state = model.init_state()
for layer in init_state:
    layer.to(Device.DEFAULT)

tokenizer = Tokenizer.from_file("20B_tokenizer.json")
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt).ids

# Initialize the model with the prompt
for token in tokens:
    #print(token)
    out, init_state = model.forward(token, init_state)

tokens.append(int(Tensor.softmax(out).argmax().numpy()))

for i in range(10):
    out, init_state = model.forward(tokens[-1], init_state)
    tokens.append(int(Tensor.softmax(out).argmax().numpy()))
    print(tokenizer.decode(tokens))