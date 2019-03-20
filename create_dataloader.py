# this function takes a datasets and then process it in away to apply transforamtions to help the model train and validate better
def create_dataloader(dataset_dir):
    train_dir = dataset_dir + '/train'
    valid_dir = dataset_dir + '/valid'
    test_dir = dataset_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
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

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
    test_datasets = datasets.ImageFolder(train_dir, transform = valid_test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)

    return trainloader, validloader, testloader
