import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

TARGET_REPS = 5
TARGET_STATE_r = False
TARGET_STATE_l = False
webcam  = True

if webcam == True:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(1)


# Curl counter variables
counter_r = 0
counter_l = 0
stage_r = None
stage_l = None


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

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

            # Get coordinates on right arm
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Get coordinates on left arm
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate angle
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            corr_r  = calculate_angle(elbow_r, shoulder_r, hip_r)
            corr_l  = calculate_angle(elbow_l, shoulder_l, hip_l)

            # Setup status box
            cv2.rectangle(image, (0, 0), (275, 120), (255, 153, 51), -1)
            cv2.rectangle(image, (1015, 0), (1280, 120), (51, 153, 255), -1)

            cv2.putText(image, str(angle_r.astype(int))+" deg",
                        (200,15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

            cv2.putText(image, str(angle_l.astype(int))+" deg",
                        (1205,15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

            if angle_r > 150:
                stage_r = "down"
            if angle_r < 30 and stage_r == "down":
                stage_r = "up"
                counter_r += 1
            if counter_r >= TARGET_REPS:
                TARGET_STATE_r = True
                
            if angle_l > 150:
                stage_l = "down"
            if angle_l < 30 and stage_l == "down":
                stage_l = "up"
                counter_l += 1
            if counter_l >= TARGET_REPS:
                TARGET_STATE_l = True

            if corr_r > 20:
                cv2.putText(image, "WARNING",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (75, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "KEEP THE RIGHT ELBOW",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (75, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "CLOSER TO YOUR BODY",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (75, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
            if corr_l > 20:
                cv2.putText(image, "WARNING",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (1090, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "KEEP THE LEFT ELBOW",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (1090, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "CLOSER TO YOUR BODY",
                            # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            (1090, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )

        except:
            pass

        # Render curl counter
        # Rep data
        cv2.putText(image, 'REPS RIGHT ARM', (15, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_r),
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'REPS LEFT ARM', (1030, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_l),
                    (1030, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255, 255, 255), 2, cv2.LINE_AA)

        if TARGET_STATE_r == True:
            cv2.putText(image, "GREAT! TASK COMPLETED!",
                        (15, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
        if TARGET_STATE_l == True:
            cv2.putText(image, "GREAT! TASK COMPLETED!",
                        (1030, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                 )

        cv2.imshow('Mediapipe Feed', image)

        if TARGET_STATE_r == True and TARGET_STATE_l == True:
            cv2.waitKey(2000)
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()