# this function creates the model and modify it based on user inputs
def create_model(arch, hidden_size):
    # TODO: Build and train your network
    if arch == 'vgg':
        model = models.vgg16(pretrained = True)
    elif arch == 'alexnet':
        model = models.vgg16(pretrained = True)
    else:
        model = models.vgg16(pretrained = True)

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088,hidden_size)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p = 0.2)),
                                            ('fc2', nn.Linear(hidden_size,102)),
                                            ('output', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier

    return model
