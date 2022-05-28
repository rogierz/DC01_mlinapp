from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def apply_mask_on_image(img, mask, out_path):
    # convert img type (if necessary)
    if not isinstance(img, (Image.Image)):
        img = tf.keras.utils.array_to_img(tf.cast(img[0] * 255.0, dtype=tf.uint8))
    
    # convert mask type (if necessary)
    if not isinstance(mask, (np.ndarray, np.generic) ):
        single_ch_mask = mask[0].numpy().copy()
    else:
        single_ch_mask = mask.copy()

    # convert one hot mask to single dim
    if single_ch_mask.shape[2] > 1:
        single_ch_mask = np.argmax(single_ch_mask, axis=2)

    # apply colormap on mask
    color_mask = (plt.cm.jet(single_ch_mask/np.max(single_ch_mask))[:,:,:3] * 255).astype(np.uint8)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # save output
    mask_img = Image.fromarray(color_mask)
    mask_img.save(os.path.join(out_path,'mask.png'))

    composite = Image.composite(img, mask_img, Image.new("L", mask_img.size, 128))
    composite.convert('RGB').save(os.path.join(out_path,'mask_on_img.png'))