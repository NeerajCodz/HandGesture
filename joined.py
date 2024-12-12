import cv2
import mediapipe as mp

def count_fingers(results):
    finger_count = 0

    # Check for thumb
    if results.landmark[4].y < results.landmark[3].y:
        finger_count += 1

    # Check for each finger
    for i in range(8, 20, 4):
        if results.landmark[i].y < results.landmark[i - 2].y:
            finger_count += 1

    return finger_count

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
                # Count fingers
                finger_count = count_fingers(hand_landmarks)

                # Display finger count
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display finger names
                finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                for i, name in enumerate(finger_names):
                    cv2.putText(frame, f"{name}: {hand_landmarks.landmark[i*4].y < hand_landmarks.landmark[i*4 - 2].y}",
                                (10, 30 + (i + 1) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
