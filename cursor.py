import cv2
import mediapipe as mp
import pyautogui

# Set up Hand tracking module from Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Get screen width and height
screen_width, screen_height = pyautogui.size()

# Set up video capture with your screen resolution
cap = cv2.VideoCapture(0)
cap.set(16, screen_width)  # Set width
cap.set(9, screen_height)  # Set height

# Smoothing parameters
smoothing_factor = 0.8
previous_x, previous_y = pyautogui.position()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[6]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Smoothly move the cursor
            x_smooth = int(previous_x * (1 - smoothing_factor) + x * smoothing_factor)
            y_smooth = int(previous_y * (1 - smoothing_factor) + y * smoothing_factor)

            # Move the cursor to the specified position
            pyautogui.moveTo(x_smooth, y_smooth)

            # Update previous position
            previous_x, previous_y = x_smooth, y_smooth

            # Draw hand landmarks and connections
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
