import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Start video capture
video = cv2.VideoCapture(0)

# Mediapipe for face detection and landmark detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load face detection and face mesh (for landmarks)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

# Label dictionary for emotions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    ret, frame = video.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Ensure the bounding box is within image bounds
            if x < 0 or y < 0 or x + w > iw or y + h > ih:
                continue  # Skip if the bounding box is invalid

            # Crop the face for emotion detection
            sub_face_img = rgb_frame[y:y + h, x:x + w]

            if sub_face_img.size == 0:
                continue  # Skip if the face region is empty

            sub_face_img_gray = cv2.cvtColor(sub_face_img, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(sub_face_img_gray, (48, 48))

            # Normalize and reshape the face image
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))

            # Predict emotion using the model
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw emotion label box above the face
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Use Face Mesh to get the landmarks
            face_mesh_results = face_mesh.process(rgb_frame)
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    # Draw facial landmarks on the face
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

                    # Example: Draw line between two key points (eyes)
                    eye_left = face_landmarks.landmark[133]  # Left eye landmark
                    eye_right = face_landmarks.landmark[362]  # Right eye landmark

                    # Convert normalized coordinates to pixel coordinates
                    eye_left_pos = (int(eye_left.x * iw), int(eye_left.y * ih))
                    eye_right_pos = (int(eye_right.x * iw), int(eye_right.y * ih))

                    # Draw a line between eyes
                    cv2.line(frame, eye_left_pos, eye_right_pos, (255, 0, 0), 2)

                    # Draw line around the mouth
                    mouth_top = face_landmarks.landmark[13]
                    mouth_bottom = face_landmarks.landmark[14]

                    mouth_top_pos = (int(mouth_top.x * iw), int(mouth_top.y * ih))
                    mouth_bottom_pos = (int(mouth_bottom.x * iw), int(mouth_bottom.y * ih))

                    cv2.line(frame, mouth_top_pos, mouth_bottom_pos, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Exit the loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
video.release()
cv2.destroyAllWindows()
