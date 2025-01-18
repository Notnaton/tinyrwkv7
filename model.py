from typing import List
from tinygrad import Tensor, TinyJit
from tinygrad.nn import GroupNorm
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.nn.state import torch_load

class RWKV:
    def __init__(self, config):
        self.n_embedding = config.get("n_embd")
        self.n_layer = config.get("n_layer")
        self.n_head = config.get("n_head")
        self.head_size = config.get("head_size")
        self.vocab_size = config.get("vocab_size")
        self.type = config.get("dtype", dtypes.float32)
        self.model = {}
        
        if "model_name" in config:
            self._load_pretrained(config["model_name"])
        else:
            self._init_weights()
    
    def _init_weights(self):
        # Initialize model weights randomly
        for i in range(self.n_layer):
            block_prefix = f'blocks.{i}.'
            att_prefix = f'{block_prefix}att.'
            ffn_prefix = f'{block_prefix}ffn.'
            
            # Attention weights
            self.model[att_prefix + 'w0'] = Tensor.uniform(self.n_embedding, self.head_size).realize()
            self.model[att_prefix + 'w1'] = Tensor.uniform(self.n_embedding, self.head_size).realize()
            self.model[att_prefix + 'w2'] = Tensor.uniform(self.head_size, self.n_embedding).realize()
            
            # Time mixing weights
            for suffix in ['x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g']:
                self.model[att_prefix + suffix] = Tensor.uniform(self.n_embedding).realize()
                
            # Other attention weights
            self.model[att_prefix + 'key.weight'] = Tensor.uniform(self.n_head * self.head_size, self.n_embedding).realize()
            self.model[att_prefix + 'value.weight'] = Tensor.uniform(self.n_head * self.head_size, self.n_embedding).realize()
            self.model[att_prefix + 'receptance.weight'] = Tensor.uniform(self.n_head * self.head_size, self.n_embedding).realize()
            self.model[att_prefix + 'output.weight'] = Tensor.uniform(self.n_embedding, self.n_head * self.head_size).realize()
            
            # Layer norms
            self.model[block_prefix + 'ln1.weight'] = Tensor.ones(self.n_embedding).realize()
            self.model[block_prefix + 'ln1.bias'] = Tensor.zeros(self.n_embedding).realize()
            self.model[block_prefix + 'ln2.weight'] = Tensor.ones(self.n_embedding).realize()
            self.model[block_prefix + 'ln2.bias'] = Tensor.zeros(self.n_embedding).realize()
            
            # FFN weights
            self.model[ffn_prefix + 'key.weight'] = Tensor.uniform(self.n_embedding * 4, self.n_embedding).realize()
            self.model[ffn_prefix + 'value.weight'] = Tensor.uniform(self.n_embedding, self.n_embedding * 4).realize()
            
        # Embedding and output layers
        self.model['emb.weight'] = Tensor.uniform(self.vocab_size, self.n_embedding).realize()
        self.model['head.weight'] = Tensor.uniform(self.vocab_size, self.n_embedding).realize()
        self.model['ln_out.weight'] = Tensor.ones(self.n_embedding).realize()
        self.model['ln_out.bias'] = Tensor.zeros(self.n_embedding).realize()

    def _load_pretrained(self, model_path):
        model = torch_load(model_path)
        
        for key in model.keys():
            if key.endswith(('att.w0', 'att.a0', 'att.a1', 'att.a2')):
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).squeeze()
            elif key.endswith('att.r_k'):
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).flatten().squeeze()
            else:
                self.model[key] = model[key].to(Device.DEFAULT).cast(self.type).squeeze()

        # Special handling for first block
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
            Tensor.zeros(self.n_embedding, dtype=self.type).to(Device.DEFAULT),
            Tensor.zeros((self.n_embedding // self.head_size, self.head_size, self.head_size), dtype=self.type).to(Device.DEFAULT),
            Tensor.zeros(self.n_embedding, dtype=self.type).to(Device.DEFAULT)
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

            xx, state[i*3+0], state[i*3+1], v_first = self._time_mixing(
                i, xx, state[i*3+0], v_first, state[i*3+1],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], 
                z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias']
            )
            x = x + xx

            xx = x.layernorm() * z[bbb+'ln2.weight'] + z[bbb+'ln2.bias']
            xx, state[i*3+2] = self._channel_mixing(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx

        x = x.layernorm() * z['ln_out.weight'] + z['ln_out.bias']
        x = z['head.weight'] @ x

        return x, state

    def _time_mixing(self, layer_id: int, x, x_prev, v_first, state,
                    x_r, x_w, x_k, x_v, x_a, x_g,
                    w0, w1, w2, a0, a1, a2, v0, v1, v2,
                    g1, g2, k_k, k_a, r_k,
                    kw, vw, rw, ow, ln_w, ln_b):
        H, N = self.n_head, self.head_size
        
        def lp_normalize(x: Tensor, dim=-1, p=2.0, eps=1e-12):
            lp_norm = (x.abs() ** p).sum(axis=dim, keepdim=True) ** (1.0 / p)
            lp_norm = lp_norm + eps
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
    
    def _channel_mixing(self, x, x_prev, x_k, kw, vw):
        xx = x_prev - x
        k = x + xx * x_k
        k = (kw @ k).relu() ** 2
        return vw @ k, x