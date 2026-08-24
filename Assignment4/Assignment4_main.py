# ============================================================
# ADS LAB - ASSIGNMENT 4
# SMART SURVEILLANCE SYSTEM
# Object Detection using YOLO + OpenCV
# Video Source: Laptop Webcam
# ============================================================

import cv2
import os
import time
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. LOAD YOLO MODEL
# ------------------------------------------------------------

model = YOLO("yolov8n.pt")


# ------------------------------------------------------------
# 2. OPEN LAPTOP WEBCAM
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)


# Check whether webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open laptop webcam.")
    exit()


# ------------------------------------------------------------
# 3. CREATE ALERT FOLDER
# ------------------------------------------------------------

ALERT_FOLDER = "alerts"

os.makedirs(ALERT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 4. CONFIDENCE THRESHOLD
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.50


# ------------------------------------------------------------
# 5. ALERT CONTROL
# ------------------------------------------------------------

last_alert_time = 0

ALERT_COOLDOWN = 5


# ------------------------------------------------------------
# 6. START VIDEO PROCESSING
# ------------------------------------------------------------

while cap.isOpened():

    # Read frame from laptop webcam
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive webcam frame.")
        break


    # --------------------------------------------------------
    # GET FRAME SIZE
    # --------------------------------------------------------

    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # DEFINE RESTRICTED ZONE
    # --------------------------------------------------------

    zone_x1 = int(width * 0.25)
    zone_y1 = int(height * 0.25)

    zone_x2 = int(width * 0.75)
    zone_y2 = int(height * 0.75)


    # --------------------------------------------------------
    # DRAW RESTRICTED ZONE
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (255, 0, 0),
        2
    )


    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )


    # --------------------------------------------------------
    # RUN YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]


    person_count = 0

    intrusion_detected = False


    # --------------------------------------------------------
    # PROCESS DETECTED OBJECTS
    # --------------------------------------------------------

    for box in results.boxes:

        # Get class ID
        class_id = int(box.cls[0])

        # Get class name
        class_name = model.names[class_id]

        # Get confidence
        confidence = float(box.conf[0])


        # ----------------------------------------------------
        # DETECT ONLY PERSON
        # ----------------------------------------------------

        if (
            class_name == "person"
            and confidence >= CONFIDENCE_THRESHOLD
        ):

            person_count += 1


            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Calculate center of person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # ------------------------------------------------
            # CHECK IF PERSON IS INSIDE RESTRICTED ZONE
            # ------------------------------------------------

            inside_zone = (
                zone_x1 <= center_x <= zone_x2
                and
                zone_y1 <= center_y <= zone_y2
            )


            # ------------------------------------------------
            # SET ALERT STATUS
            # ------------------------------------------------

            if inside_zone:

                intrusion_detected = True

                box_color = (0, 0, 255)

                label = f"INTRUSION {confidence:.2f}"

            else:

                box_color = (0, 255, 0)

                label = f"Person {confidence:.2f}"


            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                box_color,
                -1
            )


    # --------------------------------------------------------
    # DISPLAY PERSON COUNT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Persons Detected: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # INTRUSION ALERT
    # --------------------------------------------------------

    if intrusion_detected:

        current_time = time.time()


        # Save evidence image after cooldown
        if current_time - last_alert_time > ALERT_COOLDOWN:

            print("!!! INTRUSION ALERT !!!")


            # Create image filename
            filename = os.path.join(
                ALERT_FOLDER,
                f"intrusion_{int(current_time)}.jpg"
            )


            # Save evidence image
            cv2.imwrite(
                filename,
                frame
            )


            print("Evidence image saved:", filename)


            last_alert_time = current_time


        # Display intrusion message
        cv2.putText(
            frame,
            "!!! INTRUSION ALERT !!!",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


    else:

        # Normal system status
        cv2.putText(
            frame,
            "SYSTEM STATUS: NORMAL",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.imshow(
        "Smart Surveillance System",
        frame
    )


    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------------------------
# RELEASE RESOURCES
# ------------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Smart Surveillance System stopped.")