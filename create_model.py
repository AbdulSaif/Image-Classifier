# this function creates a CNN arch model and modify it based on user inputs
# including arch, hidden layer size, learning rate and numebr of epochs

def create_model(arch, hidden_size):
    '''
        This function is used to create a CNN model using required user inputs
        Arguments are: CNN arch and size of hidden layer
        Returns: a model network
    '''
    # Importing required python module
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, models, transforms

    # Defining the model arch
    if arch == 'vgg':
        model = models.vgg16(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'resnet':
        model = models.resnet18(pretrained = True)

    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False

    # Modifying classifier to match required user inputs
    from collections import OrderedDict
    if arch == 'vgg':
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hidden_size)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(p = 0.2)),
                                                ('fc2', nn.Linear(hidden_size,102)),
                                                ('output', nn.LogSoftmax(dim = 1))]))
    elif arch == 'alexnet':
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(9216,hidden_size)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(p = 0.2)),
                                                ('fc2', nn.Linear(hidden_size,102)),
                                                ('output', nn.LogSoftmax(dim = 1))]))
    else:
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hidden_size)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(p = 0.2)),
                                                ('fc2', nn.Linear(hidden_size,102)),
                                                ('output', nn.LogSoftmax(dim = 1))]))
    # storing the new classifier into the model
    model.classifier = classifier

    # returning required model by the user to be used by the main function
    return model
