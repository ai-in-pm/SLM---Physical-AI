import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device='cuda',
        checkpoint_dir='checkpoints',
        mixed_precision=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if mixed_precision else None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            texts = batch['text']  # Keep text on CPU for tokenization
            actions = batch['action'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                action_probs, _ = self.model(images, texts)
                loss = F.cross_entropy(action_probs, actions)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Calculate accuracy
            pred_actions = action_probs.argmax(dim=1)
            correct_actions += (pred_actions == actions).sum().item()
            total_actions += actions.size(0)
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': correct_actions / total_actions
            })
        
        return total_loss / len(self.train_loader), correct_actions / total_actions
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_actions = 0
        total_actions = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            texts = batch['text']
            actions = batch['action'].to(self.device)
            
            with autocast(enabled=self.mixed_precision):
                action_probs, _ = self.model(images, texts)
                loss = F.cross_entropy(action_probs, actions)
            
            pred_actions = action_probs.argmax(dim=1)
            correct_actions += (pred_actions == actions).sum().item()
            total_actions += actions.size(0)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader), correct_actions / total_actions
    
    def train(self, num_epochs, save_freq=1):
        """Full training loop"""
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Log metrics
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss = {train_loss:.4f}, '
                f'Train Acc = {train_acc:.4f}, '
                f'Val Loss = {val_loss:.4f}, '
                f'Val Acc = {val_acc:.4f}'
            )
            
            # Save checkpoint
            if epoch % save_freq == 0 or val_acc > best_val_acc:
                state = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_acc': val_acc
                }
                
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pt'
                )
                torch.save(state, checkpoint_path)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = os.path.join(
                        self.checkpoint_dir,
                        'best_model.pt'
                    )
                    torch.save(state, best_model_path)
                    
        return best_val_acc
