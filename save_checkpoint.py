# This function takes a trained model with the parameters used to train it and
# then save it in required folder as 'checkpoint.pth'
def checkpoint(save_dir, train_datasets, trained_model, arch, epochs):
    '''
        This function saves a trained network model and store it in a
        pre defined folder/location
        Arguments: save_dir, train_datasets, trained_model, arch, epochs
        Returns: saved_model
    '''
    # Importing required python module
    import torch
    # setting the path and file name of where to save the trained model
    save_path = save_dir + 'checkpoint.pth'
    trained_model.class_to_idx = train_datasets.class_to_idx
    trained_model.cpu()
    # required parameters to be saved for later training ot call of the model
    checkpoint = {'arch': arch,
                  'model_state_dict': trained_model.state_dict(),
                  'class_to_idx': trained_model.class_to_idx,
                  'epoch': epochs}
    # saving the model and its parameters in the required folder path
    saved_model = torch.save(checkpoint, save_path)
    # rertrun the saved model so that it can be stored in the main function as
    # a global variable
    return saved_model
