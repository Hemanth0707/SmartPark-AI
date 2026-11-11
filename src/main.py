from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Detect vehicles in our parking image
results = model(
    "parking.jpg",
    conf=0.15,
    imgsz=1280,
    save=True
)

# Print detected objects
for result in results:
    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"{class_name} | "
            f"confidence: {confidence:.2f}"
        )