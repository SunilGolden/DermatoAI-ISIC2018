import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def create_custom_classifier(input_features, num_classes, dropout_rate=0.1):
    classifier = nn.Sequential(
        nn.Linear(input_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        
        nn.Linear(128, num_classes)
    )
    return classifier


def create_vit_model(num_classes, pretrained=True, dropout_rate=0.1):
    # Load pre-trained ViT model
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = vit_b_16(weights=weights)

    num_features = model.heads.head.in_features
    model.heads.head = create_custom_classifier(num_features, num_classes, dropout_rate)

    return model