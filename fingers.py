import cv2
import mediapipe as mp

#thumb finger logic problem

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)  # Adjust the frame rate as needed

while cap.isOpened():
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_frame)

    # Create a blank space for finger counts
    left_finger_count = 0
    right_finger_count = 0
    left_fingers = []
    right_fingers = []

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine if it's a left or right hand
            hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb logic
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]

            if hand_label == "Left":
                is_thumb_open = thumb_tip.x < thumb_mcp.x
            else:  # Right hand
                is_thumb_open = thumb_tip.x > thumb_mcp.x

            # Count fingers
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
            finger_count = int(is_thumb_open)  # Start with the thumb state
            fingers = ["Thumb"] if is_thumb_open else []

            for tip_id in finger_tips:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    finger_count += 1
                    fingers.append(["Index", "Middle", "Ring", "Pinky"][finger_tips.index(tip_id)])

            if hand_label == "Left":
                left_finger_count = finger_count
                left_fingers = fingers
            else:
                right_finger_count = finger_count
                right_fingers = fingers

    # Display finger counts and names on the frame
    cv2.putText(frame, f"Left Fingers: {left_finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Fingers: {right_finger_count}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"{', '.join(left_fingers)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{', '.join(right_fingers)}", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
