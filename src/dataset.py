from torchvision import datasets, transforms

def get_transforms(img_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tfms, test_tfms

def get_datasets(train_dir, test_dir, img_size: int = 224):
    train_tfms, test_tfms = get_transforms(img_size)
    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    test_ds  = datasets.ImageFolder(root=str(test_dir),  transform=test_tfms)
    return train_ds, test_ds
