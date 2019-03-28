# This function is used to process an image in a specific format so that it can
# be used by a pretrained model for image classification
def process_image(image):
    ''' This function Scales, crops, and normalizes a PIL image for a PyTorch
        model usage either for training or classification
        arguments are: an image file only
        returns: an Numpy array
    '''

    # Importing required python modeules
    import numpy as np
    import torch

    # Process a PIL image for use in a PyTorch model
    # importing required PIL and Image library
    from PIL import Image as img
    # open the image for processing
    pp_image = img.open(image)
    # resize the model so that shortest side is 256 pixels
    if pp_image.size[0] < pp_image.size[1]:
        pp_image.thumbnail((256, 5000))
    else:
        pp_image.thumbnail((5000, 256))
    # cropping the image to be 224x224 pixels
    l_margin = (pp_image.width - 224) / 2
    r_margin = l_margin + 224
    b_margin = (pp_image.width - 224) / 2
    t_margin = b_margin + 224
    pp_image = pp_image.crop((l_margin, b_margin, r_margin, t_margin))
    # normalizing the image
    pp_image = np.array(pp_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    p_image = (pp_image - mean) / std

    # setting the color channel to be the first for pytorch
    p_image = p_image.transpose((2, 0, 1))

    # return the processed image so that can be used by the model
    return p_image
