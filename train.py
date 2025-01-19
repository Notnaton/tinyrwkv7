import os
import time
import random
import math
import json
from typing import List
from tinygrad import Tensor, TinyJit
from tinygrad.nn import GroupNorm, LayerNorm
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.nn.state import torch_load, safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad.nn.optim import Adam
from tokenizers import Tokenizer
import numpy as np

# --- StableMax Function ---
def stablemax(x: Tensor) -> Tensor:
    """
    StableMax Activation Function: A numerically stable alternative to Softmax.
    Applies piecewise linear scaling to logits to prevent numerical instability.
    Args:
        x (Tensor): Input tensor of logits.
    Returns:
        Tensor: Output tensor with StableMax applied.
    """
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    stable_positive = (x + 1) * positive_mask
    stable_negative = (1 / (1 - x)) * negative_mask
    scaled_x = stable_positive + stable_negative
    return scaled_x / scaled_x.sum(axis=-1, keepdim=True)

# --- RWKV_RNN Class (from rwkv7.py) ---
class RWKV_RNN:
    def __init__(self, config):
        self.n_embedding = config.get("n_embd")
        self.n_layer = config.get("n_layer")
        self.n_head = config.get("n_head")
        self.head_size = config.get("head_size")
        self.vocab_size = config.get("vocab_size")
        self.type = config.get("dtype", dtypes.float32)
        self.model = {}

        model = torch_load(config.get("model_name"))

        for key in model.keys():
            if key.endswith(('att.w0', 'att.a0', 'att.a1', 'att.a2')):
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).squeeze()
            elif key.endswith('att.r_k'):
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).flatten().squeeze()
            else:
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).squeeze()

        self.model['blocks.0.att.v0'] = self.model['blocks.0.att.a0']
        self.model['blocks.0.att.v1'] = self.model['blocks.0.att.a1']
        self.model['blocks.0.att.v2'] = self.model['blocks.0.att.a2']

        self.model['emb.weight'] = (
            model['emb.weight'].to(Device.DEFAULT).cast(self.type).layernorm()
            * model['blocks.0.ln0.weight'].to(Device.DEFAULT).cast(self.type)
            + model['blocks.0.ln0.bias'].to(Device.DEFAULT).cast(self.type)
        )

    def init_state(self):
        return [
            Tensor.zeros(self.n_embedding, dtype=self.type),
            Tensor.zeros((self.n_embedding // self.head_size, self.head_size, self.head_size), dtype=self.type),
            Tensor.zeros(self.n_embedding, dtype=self.type)
        ] * self.n_layer

    def forward(self, token: int, state: List[Tensor]):
        z = self.model
        x: Tensor = z['emb.weight'][token]
        v_first = Tensor.zeros_like(x)

        for i in range(self.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

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
            x = x + xx

            xx = x.layernorm() * z[bbb+'ln2.weight'] + z[bbb+'ln2.bias']

            xx, state[i*3+2] = self.channel_mixing(
                xx, state[i*3+2], 
                z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight']
            )
            x = x + xx

        x = x.layernorm() * z['ln_out.weight'] + z['ln_out.bias']
        x = z['head.weight'] @ x

        return x, state

    def time_mixing(self, layer_id: int, H: int, N: int, x, x_prev, v_first, state,
                    x_r, x_w, x_k, x_v, x_a, x_g,
                    w0, w1, w2, a0, a1, a2, v0, v1, v2,
                    g1, g2, k_k, k_a, r_k,
                    kw, vw, rw, ow, ln_w, ln_b):
        def lp_normalize(x: Tensor, dim=-1, p=2.0, eps=1e-12):
            # Compute the L_p norm along the specified dimension
            lp_norm = (x.abs() ** p).sum(axis=dim, keepdim=True) ** (1.0 / p)
            # Add eps to avoid divide-by-zero
            lp_norm = lp_norm + eps
            # Normalize by dividing each element by the norm
            return x / lp_norm

        xx = x_prev - x
        xr = x + xx * x_r
        xw = x + xx * x_w
        xk = x + xx * x_k
        xv = x + xx * x_v
        xa = x + xx * x_a
        xg = x + xx * x_g

        r = rw @ xr
        w = Tensor.tanh(xw @ w1) @ w2 
        k = kw @ xk
        v = vw @ xv
        a = Tensor.sigmoid(a0 + (xa @ a1) @ a2) 
        g = Tensor.sigmoid(xg @ g1) @ g2 

        if layer_id == 0:
            v_first = v
        else:
            v_gate = (xv @ v1 @ v2 + v0).sigmoid()
            v = v + (v_first - v) * v_gate

        w = w.reshape(H, N)
        k = k.reshape(H, N)
        v = v.reshape(H, N)
        a = a.reshape(H, N)

        k_k = k_k.reshape(H, N)
        kk = k * k_k
        kk = lp_normalize(kk, dim=-1, p=2.0)
        k_a = k_a.reshape(H, N)
        k = k * (1 + (a - 1) * k_a)

        w0 = w0.reshape(H, N)
        w = w0 + w
        w = (-0.606531 * w.sigmoid()).exp()

        vk = v.reshape(H, N, 1).float() @ k.reshape(H, 1, N).float()
        ab = (-kk).reshape(H, N, 1).float() @ (kk*a).reshape(H, 1, N).float()

        state = state * w.reshape(H, 1, N).float() + state @ ab.float() + vk.float()

        r = r.reshape(H, N)
        r_k = r_k.reshape(H, N)

        out = state @ r.reshape(H, N, 1)
        out = out.reshape(1, H*N)

        gn = GroupNorm(H, H*N, eps=64e-5)
        gn.weight = ln_w
        gn.bias = ln_b
        out = gn(out)
        out = out.reshape(H*N)

        direct_path = (r * k * r_k).reshape(H, N).sum(axis=1, keepdim=True) * v
        direct_path = direct_path.reshape(H*N)

        out = out + direct_path

        return (ow @ (out * g)).realize(), x.realize(), state.realize(), v_first.realize()
    
    def channel_mixing(self, x, x_prev, x_k, kw, vw):
        xx = x_prev - x
        k = x + xx * x_k
        k = (kw @ k).relu() ** 2
        return vw @ k, x
    
# --- Sampling Function (from rwkv7.py) ---
def sample_logits(logits: Tensor, temperature: float = 0.0):
    probs = stablemax(logits)
    if temperature == 0.0:
        return int(probs.argmax().numpy())
    if temperature != 1.0:
        probs = (probs ** (1.0 / temperature))
    return int(probs.multinomial().numpy().item())

# --- Config Function (from rwkv7.py) ---
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
        'model_name': file,
        'dtype': dtypes.bfloat16
    }
    return config

# --- Training Parameters ---
MODEL_PATH = 'RWKV-x070-Pile-164M-L33-D512-20241218-ctx4096.pth'
BATCH_SIZE = 2
SEQ_LEN = 1024
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01 # Removed from Adam init
GRAD_CLIP = 1.0
LOG_EVERY = 10
DATA_FILE = "Sky-T1_data_17k.json"
SAVE_EVERY = 1 # Save model every epoch
SAVE_PATH = "trained_model.safetensors"

# --- Data Loading ---
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_data(data, tokenizer, seq_len):
    tokenized_data = []
    for item in data:
        for conversation in item['conversations']:
            if conversation['from'] == 'assistant':
                text = conversation['value']
                tokens = tokenizer.encode(text).ids
                # Split into sequences of seq_len
                for i in range(0, len(tokens) - seq_len, seq_len):
                    tokenized_data.append(tokens[i:i + seq_len + 1])
                # Handle remaining tokens
                if len(tokens) % seq_len != 0 and len(tokens) > seq_len:
                    tokenized_data.append(tokens[-(seq_len+1):])
    return tokenized_data

# --- Main Training Loop ---
if __name__ == "__main__":
    config = config_from_file(MODEL_PATH)
    model = RWKV_RNN(config)
    tokenizer = Tokenizer.from_file("20B_tokenizer.json")
    vocab_size = config['vocab_size']
    
    # Load and prepare data
    json_data = load_json_data(DATA_FILE)
    data = prepare_data(json_data, tokenizer, SEQ_LEN)
    print(f"Loaded {len(data)} sequences from {DATA_FILE}")

    # Initialize optimizer
    params = [p for p in model.model.values() if isinstance(p, Tensor)]
    optimizer = Adam(params, lr=LEARNING_RATE) # Removed weight_decay
    print(f"Initialized Adam optimizer with learning rate {LEARNING_RATE}")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        total_loss = 0
        for batch_idx in range(0, len(data), BATCH_SIZE):
            batch = data[batch_idx:batch_idx + BATCH_SIZE]
            batch_loss = 0
            for seq in batch:
                state = model.init_state()
                loss = Tensor([0.0])
                for i in range(len(seq) - 1):
                    logits, state = model.forward(seq[i], state)
                    target = Tensor([seq[i+1]], dtype=dtypes.int32)
                    loss = loss + stablemax(logits).log().gather(dim=0, index=target).mul(-1)
                    print(f"Seq: {seq[i]}, Target: {seq[i+1]}, Loss: {loss.realize().numpy()}")
                batch_loss += loss.realize().numpy()
            batch_loss = batch_loss / len(batch)
            total_loss += batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient Clipping
            for p in params:
                if p.grad is not None:
                    grad_norm = p.grad.abs().max()
                    if grad_norm > GRAD_CLIP:
                        p.grad = p.grad * (GRAD_CLIP / grad_norm)
            
            optimizer.step()
            
            if (batch_idx // BATCH_SIZE) % LOG_EVERY == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx // BATCH_SIZE}, Loss: {batch_loss:.4f}")
        
        avg_loss = total_loss / (len(data) // BATCH_SIZE)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds, Avg Loss: {avg_loss:.4f}")
        
        # Save model
        if (epoch + 1) % SAVE_EVERY == 0:
            state_dict = get_state_dict(model.model)
            safe_save(state_dict, SAVE_PATH)
            print(f"Model saved to {SAVE_PATH} after epoch {epoch+1}")
    
    # Save model after training
    state_dict = get_state_dict(model.model)
    safe_save(state_dict, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH} after training")