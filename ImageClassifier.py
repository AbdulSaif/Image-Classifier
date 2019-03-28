from get_user_inputs import get_user_inputs
#from t_get_user_inputs import t_get_user_inputs
#from p_get_user_inputs import p_get_user_inputs
from create_dataloader import create_dataloader
from create_model import create_model
from save_checkpoint import checkpoint
from train import train_model
from load_checkpoint import load_checkpoint
from process_image import process_image
from predict import predict

def main():
# select which mode the user needs
    user_inputs = get_user_inputs()
    # getting user inputs to required parameters and save them to a variable
    if user_inputs.mode == 'train':
        # creating train, validate and test dataloaders
        train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader = create_dataloader(user_inputs.dir)
        # create the model to be trained
        model = create_model(user_inputs.arch, user_inputs.hidden_units)
        # training the model and save it as checkpoint
        trained_model = train_model(train_datasets, trainloader, validloader, model, user_inputs.epochs, user_inputs.learning_rate, user_inputs.device, user_inputs.save_dir)
        # save the model checkpoint
        saved_model = checkpoint(user_inputs.save_dir, train_datasets, trained_model, user_inputs.arch, user_inputs.epochs)
    else:
        # loading the trained model to use it for production
        model = create_model(user_inputs.arch, user_inputs.hidden_units)
        model = load_checkpoint(user_inputs.checkpoint_path, model,
        user_inputs.learning_rate)
        # set the model to evaluation mode
        model.eval()
        # predict class for single image
        top_prob, top_labels, top_flowers = predict(user_inputs.category_names, user_inputs.image_path, model, user_inputs.category_names, user_inputs.device)
        print(top_prob, top_labels, top_flowers)

# Call to main function to run the program
if __name__ == "__main__":
    main()
