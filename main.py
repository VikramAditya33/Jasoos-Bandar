import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Bandar ki photos
normal_img = cv2.imread("src/monkey-normal.jpg")
think_img = cv2.imread("src/monkey-thinking.jpg")
know_img = cv2.imread("src/monkey-knowing.jpg")
shocked_img = cv2.imread("src/monkey-shocked.jpg")
chill_img = cv2.imread("src/monkey-chill.jpg")
current_img = normal_img

cv2.namedWindow("Bandar Detector", cv2.WINDOW_NORMAL)

def get_finger_status(hand_landmarks):
    """Return [thumb, index, middle, ring, pinky] = 1 if up."""
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    lm = hand_landmarks.landmark

    fingers.append(1 if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x else 0)

    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]].y < lm[tip_ids[i] - 2].y else 0)

    return fingers

def get_index_tip(hand_landmarks, frame_shape):
    """Return (x, y) pixel coords of index fingertip."""
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark[8]
    return int(lm.x * w), int(lm.y * h)

def get_hand_center(hand_landmarks, frame_shape):
    """Return (x, y) pixel coords of hand center (wrist)."""
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark[0]  # Wrist landmark
    return int(lm.x * w), int(lm.y * h)


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3) as hands, \
     mp_face.FaceDetection(min_detection_confidence=0.5) as face_detector:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_result = hands.process(rgb)
        face_result = face_detector.process(rgb)

        current_img = normal_img  # default bandar

        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = get_finger_status(hand_landmarks)
                index_x, index_y = get_index_tip(hand_landmarks, frame.shape)
                hand_x, hand_y = get_hand_center(hand_landmarks, frame.shape)

                # Check finger gestures
                only_index_up = (fingers == [0, 1, 0, 0, 0])
                only_middle_up = (fingers == [0,0,1,0,0])
                only_ring_up = (fingers == [0,0,0,1,0])
                shocked_gesture = (sum(fingers) >= 3)

                # calculate distance from index to face center
                near_head = False
                chill_pose = False
                if face_result.detections:
                    detection = face_result.detections[0]
                    box = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    face_center_x = int((box.xmin + box.width / 2) * w)
                    face_center_y = int((box.ymin + box.height / 2) * h)
                    face_bottom_y = int((box.ymin + box.height) * h)  # Bottom of face
                    
                    dist = np.hypot(index_x - face_center_x, index_y - face_center_y)
                    near_head = dist < 80 
                    
                    # Check if hand is below chin (chill bandar)
                    horizontal_dist = abs(hand_x - face_center_x)
                    chill_pose = (hand_y > face_bottom_y + 20 and horizontal_dist < 150 and 
                                 sum(fingers) <= 1)
                    
                    cv2.circle(frame, (face_center_x, face_center_y), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (face_center_x, face_bottom_y), 8, (0, 255, 0), -1)  # Chin line
                    cv2.putText(frame, f"Dist:{int(dist)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Chill:{chill_pose}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Trigger different Bandar states
                if shocked_gesture:
                    current_img = shocked_img
                elif chill_pose:
                    current_img = chill_img
                elif only_index_up or only_middle_up or only_ring_up:
                    current_img = know_img
                elif near_head:
                    current_img = think_img
                else:
                    current_img = normal_img

        frame_height = 400
        frame_width = int(frame.shape[1] * frame_height / frame.shape[0])
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        monkey_resized = cv2.resize(current_img, (frame_width, frame_height))
        combined = np.hstack((frame_resized, monkey_resized))
        
        # Show combined bandar image
        cv2.imshow("Bandar Detector", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
