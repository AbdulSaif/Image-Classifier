# This function  retrieves the following 11 command line inputs from
# the user using the Argparse Python module and use them to train,
# validate and test an image classification model.

# Imports python modules
import argparse as arg

def get_user_inputs():
    """
    This function  retrieves the following 11 command line inputs from
    the user using the Argparse Python module and use them to train,
    validate and test an image classification model. If the user does not
    provide some or all of the 11 inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
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
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = arg.ArgumentParser()
    # Argument 1: The train or predict mode as --mode with default value 'train'
    parser.add_argument('--mode', type = str, default = 'train', help = 'train or predict mode')
    # Argument 2: Image Folderc to be used for training, validating and testing
    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'Image Folderc to be used for training, validating and testing')
    # Argument 3: The folder where you want to save the trained model
    parser.add_argument('--save_dir', type = str, default = 'ImageClassifier/', help = 'The folder where you want to save the trained model')
    # Argument 4: The CNN model arch to be used for creating the model
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'The CNN model arch to be used for creating the model')
    # Argument 5: The learning rate to be used in the training process
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'The learning rate to be used in the training process')
    # Argument 6: The size of the hidden layer to be used
    parser.add_argument('--hidden_units', type = float, default = 4096, help = 'The size of the hidden layer to be used')
    # Argument 7: The number of epochs to be used in the training
    parser.add_argument('--epochs', type = float, default = 5, help = 'The number of epochs to be used in the training')
    # Argument 8: The training/prediction to be done using gpu or cpu
    parser.add_argument('--device', type = str, default = 'gpu', help = 'The training/prediction to be done using gpu or cpu')
    # Argument 9: Image path that required for class prediction
    parser.add_argument('--image_path', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'Image path that required for class prediction')
    # Argument 10: The saved model path to be loaded for resuming training or for prediction
    parser.add_argument('--checkpoint_path', type = str, default = 'Checkpoint/checkpoint.pth', help = 'The saved model path to be loaded for resuming training or for prediction')
    # Argument 11: The file that will be used to convert the labels to meaningful flowers name
    parser.add_argument('--category_names', type = str, default = 'label_to_name.json', help = 'The file that will be used to convert the labels to meaningful flowers name ')

    # compile all parser argument and store it in local variable
    user_inputs = parser.parse_args()
    return user_inputs
