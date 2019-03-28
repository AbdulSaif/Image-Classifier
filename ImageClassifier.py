# Calling required modules and functions
from get_user_inputs import get_user_inputs
from create_dataloader import create_dataloader
from create_model import create_model
from save_checkpoint import checkpoint
from train import train_model
from load_checkpoint import load_checkpoint
from process_image import process_image
from predict import predict

# The heart of the program with all the training and prediction happening here
def main():
    # getting user inputs to required parameters and save them to as global variable
    user_inputs = get_user_inputs()
    # Selecting required running mode depending on user inputs to --mode
    if user_inputs.mode == 'train':
        # creating datasets and dataloaders and save them as global variables
        train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader = create_dataloader(user_inputs.dir)
        # create the required model by the user with specified --arch and hidden layer size
        model = create_model(user_inputs.arch, user_inputs.hidden_units)
        # training the model using user inputs to model arch, hidden layer size, learning rate
        # and number of required epochs and save it as global variable
        trained_model = train_model(train_datasets, trainloader, validloader, model, user_inputs.epochs, user_inputs.learning_rate, user_inputs.device, user_inputs.save_dir)
        # save the trained model so it can be called later for training or prediction
        saved_model = checkpoint(user_inputs.save_dir, train_datasets, trained_model, user_inputs.arch, user_inputs.epochs)
    else:
        # loading the trained model to use it for prediction by creating the model
        model = create_model(user_inputs.arch, user_inputs.hidden_units)
        # loading the trained model with its sprcific traits to use it for prediction
        model = load_checkpoint(user_inputs.checkpoint_path, model,
        user_inputs.learning_rate)
        # set the model to evaluation mode for faster prediction
        model.eval()
        # predict top classes (requested by user) for single image and save them as global variables
        top_prob, top_labels, top_flowers = predict(user_inputs.category_names, user_inputs.image_path, model, user_inputs.category_names, user_inputs.device, user_inputs.top_k)
        # Printing the top classes with its probabilities
        print("The top classification for this flower image are {} wtih probabilities of {}".format(top_flowers, top_prob))

# Call to main function to run the program
if __name__ == "__main__":
    main()
