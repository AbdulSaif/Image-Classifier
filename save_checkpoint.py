# this function is to save a checkpoint for the trained model
def checkpoint():
    model.class_to_idx = train_datasets.class_to_idx
    model.cpu()
    checkpoint = {'arch': user_inputs.arch,
                  'model_state_dict': model.state_dict(),
                  'criterion_state_dict': criterion.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': epochs,
                  'loss': loss}
    torch.save(checkpoint, 'checkpoint.pth')
    print("The model has been saved")
