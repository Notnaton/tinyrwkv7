from typing import List, Dict, Tuple
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import Adam
from model import RWKV
from tokenizers import Tokenizer
from tqdm import tqdm

import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Conversation:
    system: str
    exchanges: List[Tuple[str, str]]  # List of (from, value) pairs

class RWKVDataProcessor:
    def __init__(self, tokenizer_path: str):
        """Initialize the data processor with a tokenizer path."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
    def load_conversations(self, json_path: str) -> List[Conversation]:
        """Load and parse conversations from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        conversations = []
        for item in data:
            # Extract system prompt
            system_prompt = item.get('system', '')
            
            # Extract conversation exchanges
            exchanges = []
            for conv in item.get('conversations', []):
                speaker = conv.get('from', '')
                text = conv.get('value', '')
                exchanges.append((speaker, text))
                
            conversations.append(Conversation(
                system=system_prompt,
                exchanges=exchanges
            ))
            
        return conversations
    
    def prepare_training_data(self, conversations: List[Conversation]) -> List[str]:
        """Convert conversations into training sequences."""
        training_sequences = []
        
        for conv in conversations:
            # Start with system prompt if available
            if conv.system:
                sequence = conv.system + "\n\n"
            else:
                sequence = ""
                
            # Add conversation exchanges
            for speaker, text in conv.exchanges:
                if speaker == "user":
                    sequence += f"Human: {text}\n\n"
                elif speaker == "assistant":
                    sequence += f"Assistant: {text}\n\n"
                    
            training_sequences.append(sequence)
            
        return training_sequences
    
    def tokenize_sequences(self, sequences: List[str], max_length: int = None) -> List[List[int]]:
        """Tokenize sequences and optionally truncate to max_length."""
        tokenized = []
        
        for seq in sequences:
            tokens = self.tokenizer.encode(seq).ids
            if max_length:
                tokens = tokens[:max_length]
            tokenized.append(tokens)
            
        return tokenized

def create_training_batches(tokenized_sequences: List[List[int]], 
                          batch_size: int,
                          sequence_length: int = None) -> List[List[List[int]]]:
    """Create batches of sequences for training."""
    batches = []
    current_batch = []
    
    for seq in tokenized_sequences:
        if sequence_length:
            # Split long sequences into chunks of sequence_length
            for i in range(0, len(seq), sequence_length):
                chunk = seq[i:i + sequence_length]
                if len(chunk) == sequence_length:  # Only use complete chunks
                    current_batch.append(chunk)
                    if len(current_batch) == batch_size:
                        batches.append(current_batch)
                        current_batch = []
        else:
            current_batch.append(seq)
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []
    
    # Add remaining sequences if any
    if current_batch:
        batches.append(current_batch)
        
    return batches

# Example usage:
"""
# Initialize processor
processor = RWKVDataProcessor('path_to_tokenizer.json')

# Load and process data
conversations = processor.load_conversations('training_data.json')
sequences = processor.prepare_training_data(conversations)
tokenized = processor.tokenize_sequences(sequences, max_length=1024)
batches = create_training_batches(tokenized, batch_size=4, sequence_length=512)

# Use batches for training
for batch in batches:
    # Train model using batch
    pass
"""

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

class RWKVTrainer:
    def __init__(self, model: RWKV, tokenizer_path: str, learning_rate: float = 1e-4):
        self.model = model
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.optimizer = Adam([v for v in self.model.model.values()], lr=learning_rate)
        
    def prepare_batch(self, texts: List[str]) -> Tuple[List[int], List[int]]:
        # Tokenize and prepare input-target pairs
        all_tokens = [self.tokenizer.encode(text).ids for text in texts]
        max_len = max(len(tokens) for tokens in all_tokens)
        
        # Pad sequences
        padded_tokens = [tokens + [self.tokenizer.token_to_id("[PAD]")] * (max_len - len(tokens)) 
                        for tokens in all_tokens]
        
        # Create input-target pairs
        inputs = [tokens[:-1] for tokens in padded_tokens]
        targets = [tokens[1:] for tokens in padded_tokens]
        
        return inputs, targets
        
    def train_step(self, input_ids: List[int], target_ids: List[int]) -> float:
        total_loss = 0.0
        batch_size = len(input_ids)
        
        for b in range(batch_size):
            state = self.model.init_state()
            sequence_loss = Tensor.zeros(1)
            
            for t in range(len(input_ids[b])):
                logits, state = self.model.forward(input_ids[b][t], state)
                target = Tensor.zeros(self.model.vocab_size)
                target[target_ids[b][t]] = 1
                # Changed from log_softmax to using stablemax
                probs = stablemax(logits)
                loss = -(target * (probs + 1e-10).log()).sum()  # Added small epsilon for numerical stability
                sequence_loss = sequence_loss + loss
            
            # Average loss over sequence length
            sequence_loss = sequence_loss / len(input_ids[b])
            total_loss += sequence_loss.numpy()
            
            # Backward pass and optimization
            sequence_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return total_loss / batch_size
    
    def train(self, 
              train_texts: List[str],
              val_texts: List[str] = None,
              batch_size: int = 4,
              epochs: int = 10,
              save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the RWKV model on the provided texts.
        
        Args:
            train_texts: List of training text samples
            val_texts: Optional list of validation text samples
            batch_size: Number of sequences per batch
            epochs: Number of training epochs
            save_path: Optional path to save model checkpoints
            
        Returns:
            Dictionary containing training history (losses)
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            num_batches = (len(train_texts) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(train_texts), batch_size), desc=f'Epoch {epoch + 1}/{epochs}'):
                batch_texts = train_texts[i:i + batch_size]
                inputs, targets = self.prepare_batch(batch_texts)
                batch_loss = self.train_step(inputs, targets)
                train_loss += batch_loss
            
            train_loss /= num_batches
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_texts:
                val_loss = 0.0
                val_batches = (len(val_texts) + batch_size - 1) // batch_size
                
                for i in range(0, len(val_texts), batch_size):
                    batch_texts = val_texts[i:i + batch_size]
                    inputs, targets = self.prepare_batch(batch_texts)
                    
                    # Compute validation loss without gradients
                    with Tensor.no_grad():
                        batch_loss = self.train_step(inputs, targets)
                    val_loss += batch_loss
                
                val_loss /= val_batches
                history['val_loss'].append(val_loss)
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}')
            
            # Save checkpoint
            if save_path:
                checkpoint = {
                    'model_state': self.model.model,
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'history': history
                }
                Tensor.save(checkpoint, f'{save_path}/checkpoint_epoch_{epoch+1}.pth')
        
        return history