# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands

# Set up the Hands module
hands = mp_hands.Hands(
    static_image_mode=False,  # For real-time processing
    max_num_hands=2,         # Detect up to 2 hands
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variable to toggle FPS display
show_fps = True

def draw_custom_landmarks(image, hand_landmarks, connections, hand_type=""):
    """
    Draw custom landmarks and connections for the hand.
    - Circles at landmarks (joints)
    - Lines connecting the landmarks
    """
    # Define colors and sizes
    circle_color = (255, 255, 255)  # White
    line_color = (200, 200, 200)    # Light gray
    circle_radius = 3  # Smaller radius for circles
    line_thickness = 2
    text_color = (255, 0, 0)  # Blue color for landmark numbers

    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
            end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
            cv2.line(image, start_point, end_point, line_color, line_thickness)

    # Draw circles and landmark indices
    for idx, landmark in enumerate(hand_landmarks):
        cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (cx, cy), circle_radius, circle_color, -1)
        # Display the index number near the landmark
        cv2.putText(image, str(idx), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Draw custom circle with diameter between landmarks 4 and 6
    if hand_type != "":
        point1 = (int(hand_landmarks[4].x * image.shape[1]), int(hand_landmarks[4].y * image.shape[0]))
        point2 = (int(hand_landmarks[8].x * image.shape[1]), int(hand_landmarks[8].y * image.shape[0]))

        # Calculate center and diameter
        center_x = (point1[0] + point2[0]) // 2
        center_y = (point1[1] + point2[1]) // 2
        diameter = int(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

        # Draw black circle
        cv2.circle(image, (center_x, center_y), diameter // 2, (0, 0, 0), 2)  # Circle outline
        cv2.circle(image, point1, 5, (0, 255, 0), -1)  # Mark endpoint 1 in green
        cv2.circle(image, point2, 5, (0, 255, 0), -1)  # Mark endpoint 2 in green

        # Display diameter
        if hand_type == "Right":
            cv2.putText(image, f"Right Diameter: {diameter}", (image.shape[1] - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif hand_type == "Left":
            cv2.putText(image, f"Left Diameter: {diameter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image

print("Press 'q' to quit the program.")

# Initialize variables for FPS calculation
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    
    # Process the frame with MediaPipe Hands
    result = hands.process(frame_rgb)
    
    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_type = hand_info.classification[0].label  # Determine hand type (left or right)
            
            # Draw the custom landmarks and connections
            frame = draw_custom_landmarks(frame, hand_landmarks.landmark, mp_hands.HAND_CONNECTIONS, hand_type)
            
            # Display the positions of all landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                position = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                # Display the position of each landmark with new labels (L0, R0, L1, R1, etc.)
                if hand_type == "Left":
                    cv2.putText(frame, f"L{idx}: {position}", (10, 50 + 15 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                elif hand_type == "Right":
                    cv2.putText(frame, f"R{idx}: {position}", (frame.shape[1] - 180, 50 + 15 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Calculate and display FPS
    if show_fps:
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the output
    cv2.imshow('Custom Hand Tracking', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
