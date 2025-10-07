import cv2
import numpy as np
import time

# Load Haar Cascades for face and hand detection
print("Loading Haar Cascade for face and hand detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')  # Note: This may not be available in default OpenCV

# Initialize the camera
print("Initializing the camera...")
camera = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Camera initialized successfully.")
time.sleep(2)  # Allow time for the camera to initialize

# Define a function to classify actions
def classify_action(face_detected, hand_detected):
    if face_detected and hand_detected:
        action = "Working"  # Assuming the hand is near the face
    else:
        action = "Using Phone"
    
    return action

# Start reading frames and processing
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame. Retrying...")
        continue

    # Resize the frame for faster processing
    frame = cv2.flip(frame, 1)  # Flip the frame so that it acts like a mirror

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_detected = len(faces) > 0

    # Draw rectangle around detected face(s)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect hands (using a makeshift method for now, since a dedicated hand cascade isn't in OpenCV's default)
    # Hands can be roughly approximated by skin color segmentation or using Haar cascades if available
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Convert to HSV format for better skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours of possible hand regions
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hand_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Threshold to filter small areas (noise)
            hand_detected = True
            # Draw contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            break

    # Classify action based on detections
    action = classify_action(face_detected, hand_detected)

    # Display the action
    cv2.putText(frame, f"Action: {action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("Action Classification - Live Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Exit key pressed. Quitting...")
        break

# Release resources
print("Releasing the camera...")
camera.release()
print("Destroying all windows...")
cv2.destroyAllWindows()
