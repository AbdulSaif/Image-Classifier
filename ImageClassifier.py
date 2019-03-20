# getting user inputs to required parameters and save them to a variable
user_inputs = get_user_inputs()
# creating train, validate and test dataloaders
trainloader, validloader, testloader = create_dataloader(user_inputs.dir)
# create the model to be trained
model = create_model(user_inputs.arch, user_inputs.hidden_units)
# training the model and save it as checkpoint
save_path = train_model(trainloader, validloader, model, user_inputs.epochs, user_inputs.learning_rate, user_inputs.device)
