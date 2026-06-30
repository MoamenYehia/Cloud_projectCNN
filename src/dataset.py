import torch 
from torch.utils.data import Dataset , DataLoader 
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_dataloader(train_dir, test_dir , batch_size=32 , img_size=(64,64)):
    """
    Function to get the dataloaders for training and testing datasets.

    Args:
        train_dir (str): Path to the training dataset directory.
        test_dir (str): Path to the testing dataset directory.
        batch_size (int): Number of samples per batch. Default is 32.
        img_size (tuple): Desired image size (height, width). Default is (64, 64).
        """
    
    # Define transformations for the training and testing datasets
    
    train_transforms= transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize(img_size),])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size)
    ])

    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)