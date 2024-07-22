from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO
import os
import shutil
import random
import json

# Paths
source_images_dir = '/Users/benzionfisher/PycharmProjects/coin_recognition/dataset/images'
source_labels_dir = '/Users/benzionfisher/PycharmProjects/coin_recognition/dataset/labels'

train_dir = '/Users/benzionfisher/PycharmProjects/coin_recognition/train'
val_dir = '/Users/benzionfisher/PycharmProjects/coin_recognition/val'
test_dir = '/Users/benzionfisher/PycharmProjects/coin_recognition/test'

shutil.rmtree(train_dir)
shutil.rmtree(val_dir)
shutil.rmtree(test_dir)


# Create directories if they don't exist
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# Load image and label file list
image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith('.jpg')]
image_files.sort()

label_files = [f for f in os.listdir(source_labels_dir) if f.lower().endswith('.txt')]
label_files.sort()

# Ensure there is a label for each image
assert all(os.path.splitext(img)[0] + '.txt' in label_files for img in image_files), "Some images do not have corresponding labels."

train_percent = 0.7
val_percentage = 0.1
# test_percentrage = 0.1


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
    model = YOLO('yolov8n.pt')  # You can use different versions, like yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.

    model.train(
        data= cwd + '/custom_data.yaml',
        imgsz=640,
        epochs=30,
        batch=8,
        name='yolov8n_custom')


    valid_results = model.val()

    model.save('yolov8n_custom.pt')
else:
    model = YOLO('yolov8n_custom.pt')

# # Perform inference
# results = model(cwd + "/test/images/0e6c24e8-IMG_2970.jpg")
#
# # Print results
# print("Predictions:")
# for result in results:
#     # Accessing the results
#     boxes = result.boxes  # Bounding boxes
#     for box in boxes:
#         # Extracting information from each detected box
#         xyxy = box.xyxy  # Bounding box coordinates
#         cls = box.cls  # Class index
#         conf = box.conf  # Confidence score
#
#         # Print detected class names and bounding boxes
#         print(f"Detected {result.names[int(cls)]} with bounding box {xyxy} and confidence {conf}")
#
#     # Save results with bounding boxes
#     result.save()  # This will save an image with bounding boxes drawn in the same directory




# # Perform inference on the test set
test_images_path = os.path.join(test_dir, 'images')
test_labels_path = os.path.join(test_dir, 'labels')


# Perform inference on test images
results = []
for image_file in os.listdir(test_images_path):
    if image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png'):
        image_path = os.path.join(test_images_path, image_file)
        result = model(image_path)
        results.append((image_file, result))

def parse_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split()))[0] for line in lines]

ground_truth = {}
for label_file in os.listdir(test_labels_path):
    if label_file.endswith('.txt'):
        image_file = label_file.replace('.txt', '.jpg')
        label_path = os.path.join(test_labels_path, label_file)
        ground_truth[image_file] = parse_labels(label_path)


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def extract_predictions(results):
    predictions = {}
    for result in results:

        # assume one label
        _result = result[1][0]

        boxes = _result.boxes
        pred_labels = []
        for box in boxes:
            cls = int(box.cls)
            conf = box.conf
            pred_labels.append(cls)
        predictions[result[0]] = pred_labels
    return predictions

def evaluate_accuracy(ground_truth, predictions):
    y_true = []
    y_pred = []

    for img, gt_labels in ground_truth.items():
        pred_labels = predictions.get(img, [])
        if len(pred_labels) == 0:
            pred_labels = predictions.get(img.replace('.jpg', '.JPG'), [])

        y_true_sum = int(sum(gt_labels))
        pred_sum = sum(pred_labels)

        y_true.extend([y_true_sum])
        y_pred.extend([pred_sum])

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

# Extract predictions
predictions = extract_predictions(results)

# Evaluate accuracy
accuracy = evaluate_accuracy(ground_truth, predictions)
print(f"Accuracy: {accuracy}")



















#
# def load_labels(label_path):
#     labels = {}
#     for label_file in os.listdir(label_path):
#         if label_file.endswith('.txt'):
#             with open(os.path.join(label_path, label_file), 'r') as f:
#                 content = f.readlines()
#                 labels[label_file] = [line.strip() for line in content]
#     return labels
#
# def get_predictions(results):
#     predictions = {}
#     for result in results:
#
#         # assume one label
#         _result = result[0]
#
#         img_name = os.path.basename(_result.path)
#         boxes = _result.boxes
#         pred_labels = []
#         for box in boxes:
#             cls = int(box.cls)
#             conf = box.conf
#             pred_labels.append(f"{cls} {conf}")
#         predictions[img_name] = pred_labels
#     return predictions
#
# # Load ground truth labels
# ground_truth_labels = load_labels(test_labels_path)
#
# # Perform inference and get predictions
# trained = [model.predict(os.path.join(test_images_path, img)) for img in os.listdir(test_images_path)]
# predictions = get_predictions(trained)
#
# print(predictions)
#
# # Evaluate performance
# def evaluate_accuracy(ground_truth, predictions):
#     y_true = []
#     y_pred = []
#
#     for img, gt_labels in ground_truth.items():
#         try:
#             pred_labels = predictions[img.replace('.txt', '.jpg')]
#         except KeyError:
#             pred_labels = predictions[img.replace('.txt', '.JPG')]
#
#         gt_classes = [int(label.split()[0]) for label in gt_labels]
#         pred_classes = [int(label.split()[0]) for label in pred_labels]
#
#         y_true.extend(gt_classes)
#         y_pred.extend(pred_classes)
#
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
#
#     return accuracy, precision, recall, f1
#
# accurancy, precision, recall, f1 = evaluate_accuracy(ground_truth_labels, predictions)
# print(f"Accurancy: {accurancy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")



