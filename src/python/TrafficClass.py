CLASSES_LIST = [
    'car',
    'bicycle',
    'person',
    'road_sign'
]

def get_cls_dict():
    return {i: n for i, n in enumerate(CLASSES_LIST)}