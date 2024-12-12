import cv2
import mediapipe as mp

# Calibration factor (experiment with different values)
calibration_factor = -300

def calculate_hand_distance(hand_landmarks):
    # Get the 3D coordinates of the middle fingertip (assuming a right hand)
    middle_fingertip = hand_landmarks.landmark[12]
    x, y, z = middle_fingertip.x, middle_fingertip.y, middle_fingertip.z

    # Assuming a simple formula to calculate distance
    distance = int(z * calibration_factor)

    return distance

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate hand distance
                hand_distance = calculate_hand_distance(hand_landmarks)

                # Display hand distance
                cv2.putText(frame, f"Hand Distance: {hand_distance} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Distance Measurement', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
