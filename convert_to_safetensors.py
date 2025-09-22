import os
import torch
from safetensors.torch import save_file
import glob
import shutil

def convert_model_to_safetensors(model_path, output_path):
    # Delete output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f"Looking for PyTorch model files in {model_path}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the PyTorch model file
    model_files = glob.glob(os.path.join(model_path, "*.pt")) + \
                 glob.glob(os.path.join(model_path, "*.pth")) + \
                 glob.glob(os.path.join(model_path, "pytorch_model.bin"))
    
    if not model_files:
        raise FileNotFoundError(f"No PyTorch model files found in {model_path}")
    
    print(f"Found model file(s): {model_files}")
    model_file = model_files[0]  # Use the first found model file
    
    # Load the state dict
    print(f"Loading model from {model_file}")
    checkpoint = torch.load(model_file, map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    # Extract only the model weights, removing metadata
    model_state_dict = {}
    if isinstance(checkpoint, dict):
        # If model_state_dict exists in checkpoint, use that
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        # Otherwise try state_dict
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        print(f"After getting state dict - Keys available: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # Only keep tensor values
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                model_state_dict[key] = value
                print(f"Added tensor for key: {key} with shape {value.shape}")
    
    print(f"Total number of tensors to save: {len(model_state_dict)}")
    if len(model_state_dict) == 0:
        raise ValueError("No tensors found in the checkpoint! Check the model structure.")
    
    # Save as safetensors
    print(f"Converting to safetensors and saving to {output_path}")
    save_file(model_state_dict, output_path)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    # Update these paths according to your model location
    model_path = "./checkpoints"  # Path to your checkpoints directory
    output_path = "./checkpoints/model.safetensors"
    
    convert_model_to_safetensors(model_path, output_path)