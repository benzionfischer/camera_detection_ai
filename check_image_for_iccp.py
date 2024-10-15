from PIL import Image, ImageCms, PngImagePlugin
import os
import logging

# Configure logging
logging.basicConfig(filename='problematic_images.log', level=logging.INFO)

# Function to check if an image has an incorrect ICC profile
def check_image_for_iccp(image_path):
    try:
        with Image.open(image_path) as img:
            if isinstance(img, PngImagePlugin.PngImageFile):
                # Check if ICC profile exists
                iccp_profile = img.info.get('icc_profile')
                if iccp_profile:
                    # Try to apply the ICC profile
                    try:
                        ImageCms.getOpenProfile(iccp_profile)
                    except (ImageCms.PyCMSError, OSError) as e:
                        logging.info(f"Image with problematic ICC profile: {image_path}")
                        print(f"Problematic ICC profile in {image_path}")
                        return True
    except Exception as e:
        logging.info(f"Error loading image: {image_path} - {e}")
        print(f"Error with {image_path}: {e}")
        return False
    return False

# Directory to scan for images
input_folder = './train/images'

# Scan the folder and process each image
problematic_images = []
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        if check_image_for_iccp(image_path):
            problematic_images.append(image_path)

# Report problematic images
if problematic_images:
    print("\nThe following images have problematic ICC profiles:")
    i =0
    for img in problematic_images:
        print(i, img)
        i = i + 1
else:
    print("No problematic ICC profiles found.")
