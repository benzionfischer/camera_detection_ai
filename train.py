from ultralytics import YOLO
import os
import torch

# Check if MPS (Metal) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (GPU) on macOS
    print("MPS is available and will be used for computation.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS is not available, using CPU.")

cwd = os.getcwd()

# Load the pre-trained YOLOv8 model

is_to_train = True
if is_to_train:
    model = YOLO('yolov8n.pt')  # You can use different versions, like yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.

    # use GPU
    # model.to(device)

    model.train(
        data= cwd + '/custom_data.yaml',
        imgsz=640,
        epochs=150,
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
