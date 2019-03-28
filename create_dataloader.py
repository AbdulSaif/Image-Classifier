# this function takes a datasets and then apply transforamtions and store the datasets
# into three seperate datasets and dataloaders to be used for training, validation and
# testing

def create_dataloader(dataset_dir):
    # seperating the dataset into train, valid and test datasets
    train_dir = dataset_dir + '/train'
    valid_dir = dataset_dir + '/valid'
    test_dir = dataset_dir + '/test'

    # importing required python module
    import torch
    from torchvision import datasets, models, transforms

    # Define transforms for the training, validation, and testing datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder and apply transformations
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_datasets = datasets.ImageFolder(train_dir, transform = valid_test_transforms)

    # Using the datasets created, the dataloaders are created
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)

    # returing all datasets and dataloaders to be used by main function
    return train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader
