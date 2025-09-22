from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenPeerConfig:
    """Configuration class for OpenPeerLLM"""
    
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_size: int = 768  # Size of the hidden layers
    num_hidden_layers: int = 12  # Number of transformer layers
    num_attention_heads: int = 12  # Number of attention heads
    intermediate_size: int = 3072  # Size of the MLP intermediate layer
    max_position_embeddings: int = 1024  # Maximum sequence length
    layer_norm_eps: float = 1e-5  # Layer normalization epsilon
    hidden_dropout: float = 0.1  # Dropout probability for hidden layers
    attention_dropout: float = 0.1  # Dropout probability for attention layers
    
    def to_dict(self):
        """Convert the config to a dictionary"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "model_type": "openpeer_llm",
            "architectures": ["OpenPeerLLM"],
        }
        
    @classmethod
    def from_dict(cls, config_dict):
        """Create a config from a dictionary"""
        return cls(
            vocab_size=config_dict.get("vocab_size", 50257),
            hidden_size=config_dict.get("hidden_size", 768),
            num_hidden_layers=config_dict.get("num_hidden_layers", 12),
            num_attention_heads=config_dict.get("num_attention_heads", 12),
            intermediate_size=config_dict.get("intermediate_size", 3072),
            max_position_embeddings=config_dict.get("max_position_embeddings", 1024),
            layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
            hidden_dropout=config_dict.get("hidden_dropout", 0.1),
            attention_dropout=config_dict.get("attention_dropout", 0.1),
        )