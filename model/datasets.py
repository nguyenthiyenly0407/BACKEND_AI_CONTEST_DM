import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Required constants.
ROOT_DIR = os.path.join('..', 'input', 'PlantDoc-Dataset', 'train')
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
NUM_WORKERS = 4 # Number of parallel processes for data preparation.
VALID_SPLIT = 0.15 # Ratio of data for validation

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform

def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(get_train_transform(IMAGE_SIZE))
    )
    dataset_test = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(get_valid_transform(IMAGE_SIZE))
    )
    dataset_size = len(dataset)

    # Calculate the validation dataset size.
    valid_size = int(VALID_SPLIT*dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])

    return dataset_train, dataset_valid, dataset.classes
# def get_datasets(dataset_type='disease'):
#     """
#     dataset_type = 'plant' → label là loại cây
#     dataset_type = 'disease' → label là bệnh như cũ
#     """
#     if dataset_type == 'plant':
#         # Lấy tree_dict để map từ label bệnh sang loại cây
#         from class_names import tree_dict
#         label_to_plant = {}
#         for plant, diseases in tree_dict.items():
#             for disease in diseases:
#                 label_to_plant[disease] = plant

#         # Load dữ liệu gốc
#         full_dataset = datasets.ImageFolder(
#             ROOT_DIR, transform=(get_train_transform(IMAGE_SIZE))
#         )

#         # Chuyển tất cả labels về loại cây
#         plant_classes = sorted(set(label_to_plant.values()))
#         class_to_idx = {cls: idx for idx, cls in enumerate(plant_classes)}

#         def relabel_to_plant(index):
#             path, original_label = full_dataset.samples[index]
#             original_class = full_dataset.classes[original_label]
#             plant_label = label_to_plant[original_class]
#             return (path, class_to_idx[plant_label])

#         samples = [relabel_to_plant(i) for i in range(len(full_dataset))]
#         full_dataset.samples = samples
#         full_dataset.classes = plant_classes
#         full_dataset.class_to_idx = class_to_idx

#         # Tách train/val
#         dataset_test = datasets.ImageFolder(
#             ROOT_DIR, transform=(get_valid_transform(IMAGE_SIZE))
#         )
#         dataset_test.samples = samples
#         dataset_test.classes = plant_classes
#         dataset_test.class_to_idx = class_to_idx

#         indices = torch.randperm(len(full_dataset)).tolist()
#         valid_size = int(VALID_SPLIT * len(full_dataset))
#         train_indices, val_indices = indices[:-valid_size], indices[-valid_size:]

#         dataset_train = Subset(full_dataset, train_indices)
#         dataset_valid = Subset(dataset_test, val_indices)

#         return dataset_train, dataset_valid, plant_classes

#     else:
#         # === Bình thường như cũ ===
#         dataset = datasets.ImageFolder(
#             ROOT_DIR, transform=(get_train_transform(IMAGE_SIZE))
#         )
#         dataset_test = datasets.ImageFolder(
#             ROOT_DIR, transform=(get_valid_transform(IMAGE_SIZE))
#         )
#         indices = torch.randperm(len(dataset)).tolist()
#         valid_size = int(VALID_SPLIT * len(dataset))
#         train_indices, val_indices = indices[:-valid_size], indices[-valid_size:]

#         dataset_train = Subset(dataset, train_indices)
#         dataset_valid = Subset(dataset_test, val_indices)

#         return dataset_train, dataset_valid, dataset.classes


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 