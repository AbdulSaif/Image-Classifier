# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, arch, hidden_size, learning_rate):
    # initializing the model, loss criterion and optimizer
    model = create_model(user_inputs.arch, user_inputs.hidden_units)
    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

    # load the dictionary locally
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.class_to_idx = checkpoint['class_to_idx']
    return model
