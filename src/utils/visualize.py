from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def process_mask(mask):
    # convert mask type (if necessary)
    if not isinstance(mask, (np.ndarray, np.generic) ):
        single_ch_mask = mask[0].numpy().copy()
    else:
        single_ch_mask = mask.copy()

    # convert one hot mask to single dim
    if single_ch_mask.shape[2] > 1:
        single_ch_mask = np.argmax(single_ch_mask, axis=2)

    # apply colormap on mask
    den = np.max(single_ch_mask)
    if den == 0:
        den == 1
    color_mask = (plt.cm.jet(single_ch_mask/den)[:,:,:3] * 255).astype(np.uint8)

    # save output
    mask_img = Image.fromarray(color_mask)
    return mask_img

def apply_mask_on_image(img, mask, out_path, true_mask=None):
    # convert img type (if necessary)
    if not isinstance(img, (Image.Image)):
        img = tf.keras.utils.array_to_img(tf.cast(img[0] * 255.0, dtype=tf.uint8))

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # save output
    mask_img = process_mask(mask)
    mask_img.save(os.path.join(out_path,'predicted_mask.png'))

    composite = Image.composite(img, mask_img, Image.new("L", mask_img.size, 128))
    composite.convert('RGB').save(os.path.join(out_path,'predicted_mask_on_img.png'))

    if true_mask is not None:
        true_mask_img = process_mask(true_mask)
        true_mask_img.save(os.path.join(out_path,'true_mask.png'))
        
        true_composite = Image.composite(img, true_mask_img, Image.new("L", true_mask_img.size, 128))
        true_composite.convert('RGB').save(os.path.join(out_path,'true_mask_on_img.png'))
