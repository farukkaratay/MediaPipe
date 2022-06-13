import cv2  #opencv
import mediapipe as mp  #mediapipe
import numpy as np      #numpy, matris işlemleri matematik

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# 0=webcam, 1 1.kamera , "C:/Users/eren0/Pictures/Camera Roll/1.jpg"
## Setup mediapipe instance

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    a[0] > b[0]
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)


    return angle

with mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.80) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #print(frame.shape)
        width = frame.shape[0]
        height = frame.shape[1]

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                      ]

            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            hip = tuple(np.multiply(hip, [height, width]).astype(int))
            knee = tuple(np.multiply(knee, [height, width]).astype(int))
            ankle = tuple(np.multiply(ankle, [height, width]).astype(int))

            l_hip = tuple(np.multiply(l_hip, [height, width]).astype(int))
            l_knee = tuple(np.multiply(l_knee, [height, width]).astype(int))
            l_ankle = tuple(np.multiply(l_ankle, [height, width]).astype(int))

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)
            angle2 = calculate_angle(l_hip, l_knee, l_ankle)

            # Visualize angle
            angle = round(angle) #açıyı yuvarlama
            angle2 = round(angle2) #açıyı yuvarlama

            cv2.putText(image, str(angle), tuple(knee), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 150, 0), 4, cv2.LINE_AA)
            cv2.putText(image, str(angle2), tuple(l_knee), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=4))

            cv2.imshow('Test', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except:
            pass

    cap.release()
    cv2.destroyAllWindows()


