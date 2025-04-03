import cv2
import numpy as np
from screeninfo import get_monitors

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Get the primary monitor's resolution
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

print(f"Your laptop's resolution: {screen_width}x{screen_height}")

# Create a resizable window
window_name = "YOLO Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Set the window size to match the screen resolution
cv2.resizeWindow(window_name, screen_width, screen_height)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame dimensions
    height, width, channels = frame.shape

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Loop over the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:  # Confidence threshold
                # Scale the bounding box coordinates to the frame size
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                # Append the detection to the lists
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw the bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Put the label and confidence above the box
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Resize the frame to fit the screen resolution
    resized_frame = cv2.resize(frame, (screen_width, screen_height))

    # Display the resized frame
    cv2.imshow(window_name, resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()