# this function is to save a checkpoint for the trained model
def checkpoint(save_dir, train_datasets, trained_model, arch, epochs):
    import torch
    save_path = save_dir + 'checkpoint.pth'
    trained_model.class_to_idx = train_datasets.class_to_idx
    trained_model.cpu()
    checkpoint = {'arch': arch,
                  'model_state_dict': trained_model.state_dict(),
                  'class_to_idx': trained_model.class_to_idx,
                  'epoch': epochs}
    saved_model = torch.save(checkpoint, save_path)
    return saved_model
