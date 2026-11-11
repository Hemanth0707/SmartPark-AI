import cv2
import json
from ultralytics import YOLO

# Load parking image
image = cv2.imread("parking.jpg")

# Load parking slots
with open("parking_slots.json", "r") as f:
    slots = json.load(f)

# Load YOLO model
model = YOLO("yolo26s.pt")

# Vehicle classes
vehicle_classes = {"car", "truck", "bus", "motorcycle",}

# Image dimensions
height, width = image.shape[:2]

# Tile settings
tile_width = 306
tile_height = 208
overlap = 60

vehicles = []


# Detect vehicles in overlapping tiles
for y in range(0, height, tile_height - overlap):

    for x in range(0, width, tile_width - overlap):

        x2 = min(x + tile_width, width)
        y2 = min(y + tile_height, height)

        tile = image[y:y2, x:x2]

        results = model(
            tile,
            conf=0.15,
            imgsz=640,
            verbose=False
        )

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])

                if class_name in vehicle_classes:

                    bx1, by1, bx2, by2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Convert tile coordinates
                    bx1 += x
                    bx2 += x
                    by1 += y
                    by2 += y

                    vehicles.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": [bx1, by1, bx2, by2]
                    })


# Remove duplicate detections from overlapping tiles
def calculate_iou(box1, box2):

    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0

    intersection = (
        (inter_x2 - inter_x1) *
        (inter_y2 - inter_y1)
    )

    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    union = area1 + area2 - intersection

    return intersection / union


vehicles.sort(
    key=lambda v: v["confidence"],
    reverse=True
)

filtered_vehicles = []

for vehicle in vehicles:

    duplicate = False

    for existing in filtered_vehicles:

        if calculate_iou(
            vehicle["box"],
            existing["box"]
        ) > 0.5:
            duplicate = True
            break

    if not duplicate:
        filtered_vehicles.append(vehicle)


# Check whether vehicle center is inside a parking slot
def vehicle_in_slot(vehicle_box, slot):

    vx1, vy1, vx2, vy2 = vehicle_box
    sx1, sy1, sx2, sy2 = slot

    center_x = (vx1 + vx2) // 2
    center_y = (vy1 + vy2) // 2

    return (
        sx1 <= center_x <= sx2
        and
        sy1 <= center_y <= sy2
    )


# Determine occupancy
occupied = 0

print("\nDetected vehicles:", len(filtered_vehicles))
print("\nParking slot status:")

for i, slot in enumerate(slots):

    is_occupied = False

    for vehicle in filtered_vehicles:

        if vehicle_in_slot(vehicle["box"], slot):
            is_occupied = True
            break

    if is_occupied:
        status = "OCCUPIED"
        occupied += 1
    else:
        status = "FREE"

    print(f"Slot {i + 1:02d} -> {status}")


# Final statistics
total = len(slots)
free = total - occupied
occupancy = (occupied / total) * 100

print("\n==============================")
print("       PARKSENSE RESULT")
print("==============================")
print(f"Total slots : {total}")
print(f"Occupied    : {occupied}")
print(f"Free        : {free}")
print(f"Occupancy   : {occupancy:.2f}%")
print("==============================")