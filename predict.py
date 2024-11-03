import cv2
from ultralytics import YOLO

# Load your model
model = YOLO("/Users/benzionfisher/PycharmProjects/camera_detection/yolov8n_custom_200_epoches_CPU_510_images.pt")

# Run inference
test_images_path = '/Users/benzionfisher/PycharmProjects/camera_detection/test/images'
results = model.predict(source=test_images_path)

# Visualize results
for result in results:
    # Save or display each image with predictions
    result.show()  # This will display the image with bounding boxes
    # Alternatively, save results
    # result.save()  # This will save the images with bounding boxes drawn
