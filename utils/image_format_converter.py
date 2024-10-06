import os
from PIL import Image

def jpg_to_png(dir):
    # Iterate over the images and convert them
    for filename in os.listdir(dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Change from .jpg to .png, for example
            img_path = os.path.join(dir, filename)
            img = Image.open(img_path)

            # Convert and save in new format
            new_filename = os.path.splitext(filename)[0] + '.png'  # Change extension
            img.save(os.path.join(dir, new_filename))

            os.remove(img_path)
            print(f"Converted {filename} to {new_filename}")
