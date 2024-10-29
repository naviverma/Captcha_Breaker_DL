from PIL import Image
import os

# Folder containing your CAPTCHA images
captcha_folder = 'captchas'

# Dictionary to keep track of image sizes and their counts
size_count = {}

# Loop through each image and check its size
for filename in os.listdir(captcha_folder):
    if filename.endswith('.png'):
        img = Image.open(os.path.join(captcha_folder, filename))
        size = img.size  # (width, height)

        # Count occurrences of each size
        if size in size_count:
            size_count[size] += 1
        else:
            size_count[size] = 1

# Print out the image sizes and their counts
for size, count in size_count.items():
    print(f"Size {size} appears {count} times")
