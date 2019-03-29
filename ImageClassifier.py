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
    '''
        This function/program can be used to train a classification model to classify
        a datasets if supplied with required datasets. It can be used to train a model
        or use a pretrained model to classify an image file.
        Arguments: Are optional and only if user to wants the code to behave a certain
        way. These arguments includes:
        1. The train or predict mode as --mode with default value 'train'
        2. Image Folderc to be used for training, validating and testing
           as --dir with default value 'flowers/'
        3. The folder where you want to save the trained model as --save_dir
           and default value of 'ImageClassifier/'
        4. The CNN model arch to be used for creating the model as --arhc and
           the default value is 'vgg'
        5. The learning rate to be used in the training process as --learning_rate
           with default value is '0.001'
        6. The size of the hidden layer to be used as --hidden_units and
           the default value is '4096'
        7. The number of epochs to be used in the training as --epochs and
           the default value is '5'
        8. The training/prediction to be done using gpu or cpu as --device
           and the default value is 'gpu'
        9. Image path that required for class prediction as --image_path
           and the default is 'flowers/test/1/image_06743.jpg'
       10. The saved model path to be loaded for resuming training or for
           prediction as --checkpoint_path the default is 'Checkpoint/checkpoint.pth'
       11. The file that will be used to convert the labels to meaningful
           flowers name as --category_names with default value as 'label_to_name.json'
       12. The number of top classes to be displayed --top_k with defualt value of '3'
       Retruns: The model train a model or displat top class predictions of an image
    '''
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
