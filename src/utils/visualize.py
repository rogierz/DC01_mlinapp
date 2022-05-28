from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_mask_on_image(img, mask, out_path):
    # img.convert('RGB').save('../original.jpg')
    # convert one hot mask to single dim
    single_ch_mask = mask.numpy().copy()
    if single_ch_mask.shape[2] > 1:
        single_ch_mask = np.argmax(single_ch_mask, axis=2)

    color_mask = (plt.cm.jet(single_ch_mask/np.max(single_ch_mask))[:,:,:3] * 255).astype(np.uint8)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    mask_img = Image.fromarray(color_mask)
    mask_img.save(os.path.join(out_path,'mask.png'))

    composite = Image.composite(img, mask_img, Image.new("L", img.size, 128))
    composite.convert('RGB').save(os.path.join(out_path,'mask_on_img.png'))