# getting user inputs to required parameters and save them to a variable
t_user_inputs = get_t_user_inputs()
# creating train, validate and test dataloaders
trainloader, validloader, testloader = create_dataloader(t_user_inputs.dir)
# create the model to be trained
model = create_model(t_user_inputs.arch, t_user_inputs.hidden_units)
# training the model and save it as checkpoint
train_model(trainloader, validloader, model, t_user_inputs.epochs, t_user_inputs.learning_rate, t_user_inputs.device)
# loading the trained model to use it for production
checkpoint = load_checkpoint(t_user_inputs.save_dir, t_user_inputs.arch, t_user_inputs.hidden_units, t_user_inputs.learning_rate):
# set the model to evaluation mode
checkpoint.eval()
# predict class for single image
predict(image_path, checkpoint)
