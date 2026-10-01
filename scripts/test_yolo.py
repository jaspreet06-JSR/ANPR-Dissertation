from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run detection
results = model("data/uploads/test1.jpg")

# Display results
for result in results:
    result.show()

print("YOLO inference completed successfully.")