# this function takes a model, number of epochs and gpu/cpu requirments and train a model in given set of data
def train_model(trainloader, validloader, model, epochs, learning_rate, device):
    # Use GPU if it's available
    if device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # freezing model features and apply backpropagation to classifier only
    for param in model.parameters():
        param.requires_grad = False

    # defining the loss function
    criterion = nn.NLLLoss()

    # training the classifier and setting optimizer criteria
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device);

    # Training the model on the training dataset
    step = 0
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            step += 1
            # moving the inputs and labels to default device - useful if GPU is enabled
            inputs, labels = inputs.to(device), labels.to(device)
            # training starts here
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            batch_loss = 0
            test_loss = 0
            accuracy = 0
            # setting the gradients for validation to off to save memory and computations
            with torch.no_grad():
                model.eval()
                # validation starts here
                for inputs, labels in validloader:
                    # moving the inputs and labels to default device - useful if GPU is enabled
                    inputs, labels = inputs.to(device), labels.to(device)
                    output_p = model(inputs)
                    batch_loss += criterion(output_p, labels)
                    test_loss += batch_loss.item()
                    output_l = torch.exp(output_p)
                    top_p, top_class = output_l.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            # set the model in training mode again
            model.train()

            # printing statistics about how well the model is doinga
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

    # save the model checkpoint
    save_path = checkpoint()
    return save_path
