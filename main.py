from ultralytics import YOLO
import cv2
import math


# Start webcam
cap = cv2.VideoCapture(0)
cap.set(50, 840)
cap.set(60, 680)

# Model
model = YOLO("best1.pt")
print("Model loaded successfully!")

# Object classes
classNames = ["person", "Blue cow", "cow", "Boar",
              "sheep", "elephant", "Deer", "Donkey", "Fox", "Goat", "Lion",
              "Pig", "Rabbit", "Sheep", "Tiger", "antelope", "leopard"]

while True:
    success, img = cap.read()
    print("Frame read successfully!")
    results = model(source=0, show=True, conf=0.6, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Putted box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            print("Class index -->", cls)
            print("Class name -->", classNames[cls])

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()
