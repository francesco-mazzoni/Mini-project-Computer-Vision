import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

TARGET_REPS = 10
TARGET_STATE = False
webcam = True

if webcam == True:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('Project/squat.mp4')

# Curl counter variables
counter = 0
stage = None



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
            landmarks_w = results.pose_world_landmarks.landmark

            # Get real world coordinates wrt the hip reference frame
            hip_w_l = [landmarks_w[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks_w[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            heel_w_l = [landmarks_w[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                   landmarks_w[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            knee_w_l = [landmarks_w[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks_w[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            hip_w_r = [landmarks_w[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks_w[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            heel_w_r = [landmarks_w[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                        landmarks_w[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            knee_w_r = [landmarks_w[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks_w[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Get coordinates on right leg
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            heel_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z]
            f_index_r = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Get coordinates on left leg
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            heel_l = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
            f_index_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # Calculate angle
            angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            angle_r = calculate_angle(hip_r, knee_r, ankle_r)
            corr_r = calculate_angle(f_index_r, ankle_r, knee_r)
            corr_l = calculate_angle(f_index_l, ankle_l, knee_l)
            angle = min(angle_l, angle_r)
            corr_angle = min(corr_r, corr_l)

            # distances
            h_l = abs(hip_l[1] - knee_l[1])
            h_r = abs(hip_r[1] - knee_r[1])
            h_w_l = abs(hip_w_l[1] - heel_w_l[1])
            h_w_r = abs(hip_w_r[1] - heel_w_r[1])
            dist = min(h_l, h_r)
            dist_w = (h_w_l + h_w_r)/2


            # Setup status box
            cv2.rectangle(image, (0, 0), (275, 120), (255, 153, 51), -1)
            cv2.rectangle(image, (1005, 0), (1280, 120), (255, 153, 51), -1)

            cv2.putText(image, str(angle.astype(int)) + " deg",
                        (200, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )


            cv2.putText(image, "HIP HEIGHT FROM GROUND [m]",
                        (1010, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                        )

            cv2.putText(image, str(format(dist_w, '.3f')),
                        (1010, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255, 255, 255), 2, cv2.LINE_AA
                        )


            if angle > 150:
                stage = "up"
            if angle < 115 and stage == "up":
                stage = "down"
                counter += 1
            if counter >= TARGET_REPS:
                TARGET_STATE = True

            # Check if movement is correct
            if corr_angle < 115:
                cv2.putText(image, "WARNING",
                            (85, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "KEEP THE KNEE",
                            (85, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )
                cv2.putText(image, "WITHIN THE FOOT AREA",
                            (85, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 255), 1, cv2.LINE_AA
                            )

            if dist < 0.025:
                cv2.putText(image, "GREAT!! KEEP GOING!",
                            (1010, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 255, 10), 1, cv2.LINE_AA
                            )


        except:
            pass

        # Render curl counter
        # Rep data
        cv2.putText(image, 'SQUAT REPS', (15, 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255, 255, 255), 2, cv2.LINE_AA)


        if TARGET_STATE == True:
            cv2.putText(image, "GREAT! TASK COMPLETED!",
                        (15, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if TARGET_STATE == True:
            cv2.waitKey(2000)
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()