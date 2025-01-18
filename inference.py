from typing import List
from tinygrad import Tensor
from tokenizers import Tokenizer
from model import RWKV

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

def sample_logits(logits: Tensor, temperature: float = 0.0):
    probs = stablemax(logits)  # Changed from softmax to stablemax
    if temperature == 0.0:
        return int(probs.argmax().numpy())
    if temperature != 1.0:
        probs = (probs ** (1.0 / temperature))
    return int(probs.multinomial().numpy().item())

class RWKVInference:
    def __init__(self, model: RWKV, tokenizer_path: str):
        self.model = model
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        tokens = self.tokenizer.encode(prompt).ids
        state = self.model.init_state()
        
        # Initialize with prompt
        for token in tokens:
            out, state = self.model.forward(token, state)
            
        # Generate new tokens
        for _ in range(max_tokens):
            token = sample_logits(out, temperature)
            tokens.append(token)
            out, state = self.model.forward(token, state)
            
            # Early stopping on EOS token if needed
            if token == self.tokenizer.token_to_id("</s>"):
                break
                
        return self.tokenizer.decode(tokens)