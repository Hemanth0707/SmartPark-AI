import cv2
import json

image = cv2.imread("parking.jpg")

if image is None:
    print("Could not load parking.jpg")
    exit()

slots = []
drawing = False
start_point = None
current_point = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)

        x1, y1 = start_point
        x2, y2 = current_point

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            slots.append([x1, y1, x2, y2])
            print(f"Slot {len(slots)}: {[x1, y1, x2, y2]}")


cv2.namedWindow("Define Parking Slots")
cv2.setMouseCallback("Define Parking Slots", mouse_callback)

while True:

    display = image.copy()

    # Draw saved slots
    for i, (x1, y1, x2, y2) in enumerate(slots):
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Slot {i + 1}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Draw slot currently being selected
    if drawing and start_point and current_point:
        cv2.rectangle(
            display,
            start_point,
            current_point,
            (255, 0, 0),
            2
        )

    cv2.imshow("Define Parking Slots", display)

    key = cv2.waitKey(1) & 0xFF

    # S = save
    if key == ord("s"):
        with open("parking_slots.json", "w") as f:
            json.dump(slots, f, indent=4)

        print(f"\nSaved {len(slots)} parking slots.")

    # U = undo last slot
    elif key == ord("u"):
        if slots:
            removed = slots.pop()
            print(f"Removed slot: {removed}")

    # Q = quit
    elif key == ord("q"):
        break

cv2.destroyAllWindows()