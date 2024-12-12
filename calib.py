import cv2
import mediapipe as mp
import numpy as np

def calculate_hand_distance(hand_landmarks, calibration_factor):
    middle_fingertip = hand_landmarks.landmark[12]
    z = middle_fingertip.z
    distance = int(z * calibration_factor)
    return distance

def perform_calibration(num_calibration_frames):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    depth_values = []
    actual_distances = []

    print(f"Performing calibration for {num_calibration_frames} frames.")
    print("Place a reference object (e.g., a ruler) in the view of the camera.")

    frame_count = 0
    while frame_count < num_calibration_frames:
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
                # Capture depth value
                middle_fingertip = hand_landmarks.landmark[12]
                z = middle_fingertip.z
                depth_values.append(z)

                # Measure actual distance (you need to input this manually)
                actual_distance = float(input(f"Enter actual distance for frame {frame_count + 1} (in cm): "))
                actual_distances.append(actual_distance)

                frame_count += 1

                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Calibration', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to finish calibration
            break

    cap.release()
    cv2.destroyAllWindows()

    # Perform linear regression to find calibration factor
    calibration_factor = np.polyfit(depth_values, actual_distances, 1)[0]

    print(f"Calibration Factor: {calibration_factor}")
    return calibration_factor

def main():
    num_calibration_frames = 5  # Adjust this number based on your preference
    calibration_factor = perform_calibration(num_calibration_frames)

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
                # Calculate hand distance using the calibrated factor
                hand_distance = calculate_hand_distance(hand_landmarks, calibration_factor)

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
