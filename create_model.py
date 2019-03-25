# this function creates the model and modify it based on user inputs
def create_model(arch, hidden_size):

    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, models, transforms

    # TODO: Build and train your network
    if arch == 'vgg':
        model = models.vgg16(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained = True)

    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False

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

    model.classifier = classifier

    print("Model created")
    return model
