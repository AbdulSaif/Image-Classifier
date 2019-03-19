#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Abdul.Saif
# DATE CREATED: 25/01/2019
# REVISED DATE:
# PURPOSE: Create a function that retrieves the following 3 command line inputs
#          from the user using the Argparse Python module. If the user fails to
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse as arg

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument
#       collection that you created with this function
#
def get_user_inputs():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 3 command line arguments. If
    the user fails to provide some or all of the 3 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = arg.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: folder's path
    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'flower images folder\'s path')
    # Argument 2: checkpoint folder's path
    parser.add_argument('--save_dir', type = str, default = 'ImageClassifier/', help = 'Where to save the checkpoint - folder\'s path')
    # Argument 4: Which CNN model architecture to be used
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'Which CNN model architecture to be used')
    # Argument 5: the the learning rate you need to use to train the model
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate to train the selected model')
    # Argument 6: the size of hiddden layer to be used in the classifier
    parser.add_argument('--hidden_units', type = float, default = 512, help = 'size of hiddden layer to be used in the classifier')
    # Argument 7: the size of epochs to be used for training
    parser.add_argument('--epochs', type = float, default = 5, help = 'the size of epochs to be used for training')
    # Argument 8: is gpu required to train the model
    parser.add_argument('--gpu', type = str, 'cpu' = 0.01, help = 'is gpu required to train the model')

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    user_inputs = parser.parse_args()

    return user_inputs
