# This function takes a file that has labels and flowers name reads it and it
# then returns it as dictionary containing labels as index and flowers name as
# values

def label_mapping(filename):
    # importing required python module to read .json file
    import json

    with open(filename, 'r') as f:
        label_to_name = json.load(f)
    # returns dictionary containing labels as index and flowers name as values
    # so that can be used by the main function
    return label_to_name
