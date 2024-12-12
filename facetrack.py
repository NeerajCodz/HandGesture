# Import necessary libraries
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variable to toggle FPS display
show_fps = True

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

    # Process the frame with MediaPipe Face Mesh
    result = face_mesh.process(frame_rgb)

    # If face landmarks are detected
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw the face mesh
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                start = face_landmarks.landmark[start_idx]
                end = face_landmarks.landmark[end_idx]

                # Convert normalized coordinates to pixel values
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))

                # Draw white lines for mesh connections
                cv2.line(frame, start_point, end_point, (255, 255, 255), 1)

            # Draw white dots for landmarks
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    # Calculate and display FPS
    if show_fps:
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the output
    cv2.imshow('Face Mesh Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
