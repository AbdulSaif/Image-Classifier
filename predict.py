def predict(filename, image_path, model, category_names, device, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    from process_image import process_image
    import torch
    from label_mapping import label_mapping

    category_names = label_mapping(filename)

    # TODO: Implement the code to predict the class from an image file
    # process the image so that it can be fit in the model
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze_(0)
    # run the image in the model to get log of probabilities
    log_prob = model(image)
    # convert the log into actual probabilities
    prob = torch.exp(log_prob)
    # get the top 5 probabilities
    top_prob, top_lab = prob.topk(topk) # get the top 5 results
    top_prob = top_prob.detach().numpy().tolist()[0]
    top_lab = top_lab.detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_lab]
    top_flowers = [category_names[idx_to_class[lab]] for lab in top_lab]
    return top_prob, top_labels, top_flowers
