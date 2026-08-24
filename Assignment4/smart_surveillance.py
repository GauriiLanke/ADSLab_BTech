from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera.")
        break

    # Run YOLO detection
    results = model(
        frame,
        verbose=False
    )

    # Draw YOLO detections
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow(
        "YOLO Object Detection",
        annotated_frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()