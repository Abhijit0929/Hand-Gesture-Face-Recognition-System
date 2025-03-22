import cv2
import mediapipe as mp
import pyttsx3

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detect_gesture(landmarks):
    """Detect different hand gestures based on finger positions."""
    thumb_tip = landmarks[4]
    thumb_base = landmarks[1]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_base = landmarks[5]  # Base of index finger
    pinky_base = landmarks[17]  # Base of pinky

    # Gesture: Thumbs Up 👍
    if thumb_tip.y < thumb_base.y and index_tip.y > thumb_base.y and middle_tip.y > thumb_base.y:
        return "Thumbs Up"

    # Gesture: Thumbs Down 👎
    if thumb_tip.y > thumb_base.y and index_tip.y < thumb_base.y and middle_tip.y < thumb_base.y:
        return "Thumbs Down"

    # Gesture: Open Palm ✋ (All fingers extended)
    if (index_tip.y < index_base.y and 
        middle_tip.y < index_base.y and 
        ring_tip.y < index_base.y and 
        pinky_tip.y < pinky_base.y):
        return "Open Palm"

    # Gesture: Fist ✊ (All fingers curled)
    if (index_tip.y > index_base.y and 
        middle_tip.y > index_base.y and 
        ring_tip.y > index_base.y and 
        pinky_tip.y > pinky_base.y):
        return "Fist"

    # Gesture: Peace Sign ✌ (Index and Middle Finger up, others down)
    if (index_tip.y < index_base.y and 
        middle_tip.y < index_base.y and 
        ring_tip.y > index_base.y and 
        pinky_tip.y > pinky_base.y):
        return "Peace Sign"

    # Gesture: Pointing One Finger ☝ (Index finger up, others down)
    if (index_tip.y < index_base.y and 
        middle_tip.y > index_base.y and 
        ring_tip.y > index_base.y and 
        pinky_tip.y > pinky_base.y):
        return "Pointing Up"

    return None  # No recognized gesture

# Initialize Camera
cap = cv2.VideoCapture(0)

print("Show different hand gestures to get a response!")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            gesture = detect_gesture(hand_landmarks.landmark)
            if gesture:
                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                speak(gesture)

    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
