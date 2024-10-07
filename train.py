from ultralytics import YOLO
import os
import shutil
import random
from utils import image_format_converter
import tensorflow as tf

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)



# Paths
source_images_dir = '/Users/benzionfisher/PycharmProjects/camera_detection/dataset/images'
source_labels_dir = '/Users/benzionfisher/PycharmProjects/camera_detection/dataset/labels'

train_dir = '/Users/benzionfisher/PycharmProjects/camera_detection/train'
val_dir = '/Users/benzionfisher/PycharmProjects/camera_detection/val'
test_dir = '/Users/benzionfisher/PycharmProjects/camera_detection/test'


# verify all images are in '.png' format
image_format_converter.jpg_to_png(source_images_dir)

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)

if os.path.isdir(val_dir):
    shutil.rmtree(val_dir)

if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)


# Create directories if they don't exist
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Load image and label file list
image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.png')]
image_files.sort()

for f in os.listdir(source_images_dir):
    if not(f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.png')):
        print(f'not end with ____: {f.lower()}')

label_files = [f for f in os.listdir(source_labels_dir) if f.lower().endswith('.txt')]
label_files.sort()

# Ensure there is a label for each image
assert all(os.path.splitext(img)[0] + '.txt' in label_files for img in image_files), "Some images do not have corresponding labels."

train_percent = 0.7
val_percentage = 0.1
# test_percentrage = 0.2


# Shuffle and split data
random.shuffle(image_files)
total = len(image_files)

train_size = int(total * train_percent)
val_size = int(total * val_percentage)
test_size = int(total - train_size - val_size)

# Make sure sizes match the required counts
assert train_size + val_size + test_size == total, "Total size does not match the required splits."

train_images = image_files[:train_size]
val_images = image_files[train_size:train_size + val_size]
test_images = image_files[train_size + val_size:]

def move_files(image_list, dest_images, dest_labels):
    for image in image_list:
        label = os.path.splitext(image)[0] + '.txt'
        shutil.copy(os.path.join(source_images_dir, image), os.path.join(dest_images, image))
        shutil.copy(os.path.join(source_labels_dir, label), os.path.join(dest_labels, label))

move_files(train_images, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
move_files(val_images, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
move_files(test_images, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

print("Files have been successfully moved.")

cwd = os.getcwd()


# Load the pre-trained YOLOv8 model

is_to_train = True
if is_to_train:
    model = YOLO('yolov8n_custom.pt')  # You can use different versions, like yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.

    model.train(
        data= cwd + '/custom_data.yaml',
        imgsz=640,
        epochs=40,
        batch=8,
        name='yolov8n_custom')

    model.save('yolov8n_custom.pt')
else:
    model = YOLO('yolov8n_custom.pt')


# Evaluate the model on the test set
metrics = model.val(data=cwd + '/test_data.yaml', imgsz=640)

# Print out the metrics
print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.2f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.2f}")
print(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.2f}")
print(f"mAP@0.5:0.95: {metrics.results_dict['metrics/mAP50(B)']:.2f}")
