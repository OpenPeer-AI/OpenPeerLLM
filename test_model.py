import os
import sys
import torch
from typing import List, Dict

def test_tokenizer():
    print("Testing tokenizer...")
    from src.tokenization_openpeer import OpenPeerTokenizer
    
    tokenizer = OpenPeerTokenizer()
    test_text = "Hello world"
    
    tokens = tokenizer(test_text)
    print(f"Input text: {test_text}")
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens["input_ids"])
    print(f"Decoded: {decoded}")
    
def test_model_config():
    print("\nTesting model configuration...")
    from src.configuration_openpeer import OpenPeerConfig
    
    config = OpenPeerConfig()
    print("Model Configuration:")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Number of Layers: {config.num_hidden_layers}")
    print(f"Number of Attention Heads: {config.num_attention_heads}")
    
def test_model_architecture():
    print("\nTesting model architecture...")
    from src.modeling_openpeer import OpenPeerLLM
    from src.configuration_openpeer import OpenPeerConfig
    
    config = OpenPeerConfig()
    model = OpenPeerLLM(config)
    
    # Print model structure
    print("Model Structure:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def run_inference_test():
    print("Initializing OpenPeerLLM...")
    from src.modeling_openpeer import OpenPeerLLM
    from src.configuration_openpeer import OpenPeerConfig
    from src.tokenization_openpeer import OpenPeerTokenizer

    config = OpenPeerConfig()
    model = OpenPeerLLM(config)
    tokenizer = OpenPeerTokenizer()
    
    # Test cases
    test_prompts = [
        "Explain how decentralized computing works.",
        "What are the benefits of peer-to-peer networks?",
        "How does distributed machine learning improve model training?"
    ]
    
    print("\nRunning inference tests...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        try:
            # Tokenize input
            inputs = tokenizer(prompt)
            input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long)
            
            # Run model
            outputs = model(input_ids)
            
            # Get predictions
            logits = outputs["logits"]
            predictions = torch.argmax(logits[0], dim=-1)
            response = tokenizer.decode(predictions.tolist())
            
            print(f"Response: {response}")
            print("-" * 80)
        except Exception as e:
            print(f"Error during inference: {str(e)}")
    
    # Test model properties
    print("\nModel Architecture:")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Number of Layers: {model.config.num_hidden_layers}")
    print(f"Number of Attention Heads: {model.config.num_attention_heads}")
    
    # Memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print("\nTest completed!")

def main():
    print("Starting OpenPeerLLM tests...")
    print("=" * 80)
    
    try:
        test_tokenizer()
    except Exception as e:
        print(f"Tokenizer test failed: {str(e)}")
        
    try:
        test_model_config()
    except Exception as e:
        print(f"Config test failed: {str(e)}")
        
    try:
        test_model_architecture()
    except Exception as e:
        print(f"Model architecture test failed: {str(e)}")
        
    print("=" * 80)
    print("Tests completed!")

    try:
        run_inference_test()
    except Exception as e:
        print(f"Inference test failed: {str(e)}")

if __name__ == "__main__":
    main()