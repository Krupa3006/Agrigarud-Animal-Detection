from ultralytics import YOLO
import cv2
import math

# Model
model = YOLO("best1.pt")
print("Model loaded successfully!")

# Object classes
classNames = ["Blue cow", "cow", "Boar",
              "sheep", "elephant", "Deer", "Donkey", "Fox", "Goat", "Lion",
              "Pig", "Rabbit", "Sheep", "Tiger", "antelope", "leopard"]

# Image path
image_path = "test/WhatsApp Image 2024-04-26 at 00.19.48_6bb45fb4.jpg"

# Read image
img = cv2.imread(image_path)
# if img is None:
#     print("Error loading image")
#     exit()

# Perform object detection
results = model(img,show=True, conf=0.6, stream=True)
for r in results:
    boxes = r.boxes
for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # putted box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Add a test rectangle to the frame
            cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class index -->", cls)
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
# cv2.imshow('Image', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()