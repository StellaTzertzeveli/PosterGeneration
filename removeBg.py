"""this class is passed a single image with its label
then rembg turns the background black and through open cv the black is turned into transparent
the outcome of the class is a png transparent image"""

from io import BytesIO
from rembg import remove
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time


class RemoveBackground:

    def __init__(self, image):
        self.pose_with_bg = Image.open(image)

    def remove_background(self):
        # Remove the background
        output_img = remove(self.pose_with_bg)

        # Convert the PIL image to a NumPy array (RGB format)
        output_array = np.array(output_img)

        # Convert RGB to BGR for OpenCV
        final_bl_img = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        return final_bl_img

    def show_bl_img(self, final_bl_img):
        # Display the image using matplotlib
        final_img_rgb = cv2.cvtColor(final_bl_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
        plt.imshow(final_img_rgb)
        plt.title("Output Image")
        plt.axis('off')  # Hide axes
        plt.show()


    def final_trans_img(self, final_bl_img, label):
        """if the bgr channels are all black/zero then make black_mask true.
        so make the alpha channel 255 (fully transparent) otherwise zero.
        Then it adds that channel onto the new image."""

        # convert the black background to transparent
        black_mask = np.all(final_bl_img == 0, axis=-1)
        alpha = np.uint8(np.logical_not(black_mask)) * 255

        # Stack arrays in sequence depth wise
        bgra = np.dstack((final_bl_img, alpha))

        # convert to rgb for saveing but keep bg trans
        fixed_colors = bgra.copy()
        fixed_colors[..., :3] = cv2.cvtColor(fixed_colors[..., :3], cv2.COLOR_BGR2RGB)

        # Save the image with transparent background in no_bg_images folder
        filename = os.path.join("no_bg_images", f"{int(time.time())}_{label}.png")
        cv2.imwrite(filename, bgra)
        return bgra, fixed_colors

    def show_final_img(self, bgra):
        # !!! plt used RGB but cv2 uses BGR
        # Display the final image
        plt.imshow(bgra)
        plt.title("final trans img")
        plt.axis('off')
        plt.show()

