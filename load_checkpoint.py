# This function loads a pretrained model for further future use either for
# training or image classiifcation usage
def load_checkpoint(filepath, arch, hidden_size, learning_rate):
    # initializing the model with its CNN arch, hidden layer size etc..
    model = create_model(user_inputs.arch, user_inputs.hidden_units)

    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False

    # load the dictionary locally
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.class_to_idx = checkpoint['class_to_idx']

    # returns the pretrained loaded model
    return model
