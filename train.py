# This function is used to train a model in classifying a set of data using a
# pre-defined parameters of CNN arch, learning rate, epochs and size of hidden
# layer

def train_model(train_datasets, trainloader, validloader, model, epochs, learning_rate, device, save_dir):
    # Importing required python modules
    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, models, transforms

    # Use GPU if it's available and required by the user
    if device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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
        count = 1
        for inputs, labels in trainloader:
            print(count)
            count += 1
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
                count = 1
                # validation starts here
                for inputs, labels in validloader:
                    print(count)
                    count += 1
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
            test_losses.append(test_loss/len(validloader))
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
    # returning the trained model to main function so it can be stored as a
    # global variable
    return model
