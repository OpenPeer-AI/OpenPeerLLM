import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from src.modeling_openpeer import OpenPeerLLM
from src.configuration_openpeer import OpenPeerConfig
from src.tokenization_openpeer import OpenPeerTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, 
                               truncation=True,
                               max_length=self.max_length)
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Create labels for causal LM (shifted input_ids)
        labels = input_ids[1:] + [self.tokenizer.eos_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored in loss
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_path,
    log_interval=100
):
    model.train()
    total_steps = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                }, f"{save_path}/best_model.pt")
                
        # Save checkpoint
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
        }
        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch+1}.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--load_checkpoint", type=str, help="Path to model checkpoint to continue training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    config = OpenPeerConfig()
    model = OpenPeerLLM(config).to(device)
    tokenizer = OpenPeerTokenizer()
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        
    # Load training data
    print("Loading training data...")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
        
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * args.num_epochs)
    
    # Load optimizer state if resuming training
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Train the model
    print("Starting training...")
    train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path,
    )
    
if __name__ == "__main__":
    main()