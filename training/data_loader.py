import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split='train', max_text_length=128):
        """
        Dataset for multimodal training data
        Args:
            data_dir (str): Directory containing images and annotations
            split (str): 'train' or 'val'
            max_text_length (int): Maximum text sequence length
        """
        self.data_dir = data_dir
        self.split = split
        self.max_text_length = max_text_length
        
        # Load annotations
        with open(os.path.join(data_dir, f'{split}_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
            
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        """
        Get a single training example
        Returns:
            dict: Contains 'image', 'text', and 'action' tensors
        """
        ann = self.annotations[idx]
        
        # Load and preprocess image
        image_path = os.path.join(self.data_dir, 'images', ann['image_file'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        
        # Get text instruction and target action
        text_instruction = ann['instruction']
        target_action = torch.tensor(ann['action_label'], dtype=torch.long)
        
        return {
            'image': image_tensor,
            'text': text_instruction,
            'action': target_action
        }

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create training and validation dataloaders
    """
    train_dataset = MultimodalDataset(data_dir, split='train')
    val_dataset = MultimodalDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
