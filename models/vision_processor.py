import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

class LightweightVisionProcessor(nn.Module):
    def __init__(self, feature_dim=512, efficient_net_version='b0'):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Initialize EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{efficient_net_version}')
        
        # Remove the original classifier
        self.backbone._fc = nn.Identity()
        
        # Add a lightweight feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, feature_dim),  # EfficientNet-b0 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, images):
        """
        Process images through the vision backbone and project to feature space
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
        Returns:
            torch.Tensor: Visual features [B, feature_dim]
        """
        features = self.backbone.extract_features(images)
        features = torch.mean(features, dim=(2, 3))  # Global average pooling
        return self.feature_projection(features)
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for the model
        Args:
            image (PIL.Image): Input image
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.preprocess(image).unsqueeze(0)  # Add batch dimension
