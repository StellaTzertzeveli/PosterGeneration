from io import BytesIO
from rembg import remove
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the input image
input_path = "fish.webp"
with open(input_path, "rb") as input_file:
    input_data = input_file.read()

# Remove the background
output_data = remove(input_data)

# Convert the output to a PIL image
output_image = Image.open(BytesIO(output_data))

# Convert the PIL image to a NumPy array (RGB format)
output_array = np.array(output_image)

# Convert RGB to BGR for OpenCV
output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

# Display the image using matplotlib
output_array_rgb = cv2.cvtColor(output_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
plt.imshow(output_array_rgb)
plt.title("Output Image")
plt.axis('off')  # Hide axes
plt.show()

"""if the bgr channels are all black/zero then make black_mask true. 
so make the alpha channel 255 (fully transparent) otherwise zero. 
Then it adds that channel onto the new image."""

# convert the black background to transparent
black_mask = np.all(output_array_rgb == 0, axis=-1)
alpha = np.uint8(np.logical_not(black_mask)) * 255

# Stack arrays in sequence depth wise
bgra = np.dstack((output_array_rgb, alpha))


# convert to rgb for saveing but keep bg trans
fixed_colors = bgra.copy()
fixed_colors[..., :3] = cv2.cvtColor(fixed_colors[..., :3], cv2.COLOR_BGR2RGB)
cv2.imwrite("fix.png", fixed_colors)

# !!! plt used RGB but cv2 uses BGR
# Display the final image
plt.imshow(bgra)
plt.title("final trans img")
plt.axis('off')
plt.show()
