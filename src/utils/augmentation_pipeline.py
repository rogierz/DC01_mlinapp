import albumentations as A

def get_transformation_pipeline(augmentation=[]):
    augmentation_pipeline = []
    for t, p in augmentation:
        transformation = A.OneOf([
            t
        ],p = p)
        augmentation_pipeline.append(transformation)
    
    return A.Compose(augmentation_pipeline)