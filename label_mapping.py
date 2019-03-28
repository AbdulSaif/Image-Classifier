def label_mapping(filename):
    import json

    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
