# select which mode the user needs
mode = get_mode_user_inputs()
# getting user inputs to required parameters and save them to a variable
if mode == 'train'
    user_inputs = t_get_user_inputs()
    # creating train, validate and test dataloaders
    trainloader, validloader, testloader = create_dataloader(user_inputs.dir)
    # create the model to be trained
    model = create_model(user_inputs.arch, user_inputs.hidden_units)
    # training the model and save it as checkpoint
    train_model(trainloader, validloader, model, user_inputs.epochs, user_inputs.learning_rate, user_inputs.device)
else:
    user_inputs = p_get_user_inputs()
    # loading the trained model to use it for production
    checkpoint = load_checkpoint(user_inputs.checkpoint_path, user_inputs.arch, user_inputs.hidden_units, user_inputs.learning_rate):
    # set the model to evaluation mode
    checkpoint.eval()
    # predict class for single image
    predict(user_inputs.image_path, user_inputs.checkpoint_path, user_inputs.category_names, user_inputs.device)
