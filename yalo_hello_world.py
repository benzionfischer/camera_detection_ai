from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use different versions, like yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.

# Load an image (you can replace 'path/to/your/image.jpg' with an actual image path)
image_path = '/Users/benzionfisher/Desktop/' + 'coins.jpg'

# Perform inference
results = model(image_path)

# Print results
print("Predictions:")
for result in results:
    # Accessing the results
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        # Extracting information from each detected box
        xyxy = box.xyxy  # Bounding box coordinates
        cls = box.cls  # Class index
        conf = box.conf  # Confidence score

        # Print detected class names and bounding boxes
        print(f"Detected {result.names[int(cls)]} with bounding box {xyxy} and confidence {conf}")

    # Save results with bounding boxes
    result.save()  # This will save an image with bounding boxes drawn in the same directory
