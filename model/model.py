from torchvision import models

import torch.nn as nn

def model_config(model_name='resnet18'):
    model = {
        'efficientnetb0': models.efficientnet_b0(weights='DEFAULT'),
        'resnet18': models.resnet18(weights='DEFAULT'),
        'resnet50': models.resnet50(weights='DEFAULT')
    }
    return model[model_name]

def build_model(model_name='efficientnetb0', fine_tune=True, num_classes=10):
    model = model_config(model_name)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    if model_name == 'efficientnetb0':
        model.classifier[1] = nn. Linear(in_features=1280, out_features=num_classes)
    if model_name == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=num_classes)
    if model_name == 'resnet50':
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
    return model