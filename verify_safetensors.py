from safetensors.torch import load_file
import os

def verify_safetensors(file_path):
    print(f"Loading safetensors file from {file_path}")
    try:
        state_dict = load_file(file_path)
        print(f"Successfully loaded {len(state_dict)} tensors")
        print("\nSample of tensors:")
        for key in list(state_dict.keys())[:5]:
            print(f"{key}: {state_dict[key].shape}")
        return True
    except Exception as e:
        print(f"Error loading safetensors file: {e}")
        return False

if __name__ == "__main__":
    file_path = "./checkpoints/model.safetensors"
    verify_safetensors(file_path)