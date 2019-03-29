# This function loads a pretrained model for further future use either for
# training or image classiifcation usage
def load_checkpoint(filepath, model, learning_rate):
    '''
        This function load a pretrained model and initialize it using
        previous optimized weights for further usage (further training
        or prediction).
        Arguments: filepath, model, learning_rate
        Returns: loaded model
    '''
    # Importing required python modules
    from create_model import  create_model
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, models, transforms

    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False

    # load the dictionary locally
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model.class_to_idx = checkpoint['class_to_idx']

    # returns the pretrained loaded model
    return model
