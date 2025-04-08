import matplotlib.pyplot as plt
import numpy as np
import cv2

# get image info
image = cv2.imread('DATA/UsainBolt/pose (1).jpg')
print('This image is:', type(image),
      ' with dimensions:', image.shape)

image_copy = np.copy(image)
# Resize the image
resized_image = cv2.resize(image_copy, (1000, 900))

# convert the colorspace
img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# define the color threshold
lower_blue = np.array([0,0,230])
upper_blue = np.array([250,250,255])

# create a mask
mask = cv2.inRange(img, lower_blue, upper_blue)

# turn bg into black
masked_image = np.copy(img)
masked_image[mask != 0] = [0, 0, 0]

# mask and add a background image
background_image = cv2.imread('bg-test.png')
# Resize the image
resized_bg = cv2.resize(background_image, (1000, 900))
bg = cv2.cvtColor(resized_bg, cv2.COLOR_BGR2RGB)
crop_background = bg[0:900, 0:1000]

crop_background[mask == 0] = [0,0,0]

# combine the image with black bg and new bg
complete_image = masked_image + crop_background
plt.imshow(complete_image)
cv2.imshow("cropped Image", complete_image)

# keep the window open
cv2.waitKey(0)
cv2.destroyAllWindows()