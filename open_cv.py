import cv2 as cv
import face_recognition
import mediapipe as mp
import os
import pyttsx3
import pickle
import speech_recognition as sr

# Load the default face (abhi.jpg)
known_image_path = "abhi.jpg"
if not os.path.exists(known_image_path):
    raise FileNotFoundError(f"Known image file '{known_image_path}' not found.")

known_image = face_recognition.load_image_file(known_image_path)
known_faces = face_recognition.face_encodings(known_image)

if len(known_faces) > 0:
    known_face_encodings = [known_faces[0]]  # Store as the first known face
    known_face_names = ["Abhi"]  # Default name
    print("Known face (Abhi) loaded successfully.")
else:
    print("No faces detected in the known image.")
    exit()

# Try loading previously stored faces from file
try:
    with open("known_faces.pkl", "rb") as f:
        stored_encodings, stored_names = pickle.load(f)
        known_face_encodings.extend(stored_encodings)
        known_face_names.extend(stored_names)
except FileNotFoundError:
    print("No additional faces stored yet.")

# Load YOLO model for object detection
weights_path = "yolov4.weights"
cfg_path = "yolov4.cfg"
names_path = "coco.names"

for path in [weights_path, cfg_path, names_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLO file '{path}' not found.")

net = cv.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

if len(unconnected_layers.shape) == 1:
    output_layers = [layer_names[int(unconnected_layers[0]) - 1]]
else:
    output_layers = [layer_names[int(i[0]) - 1] for i in unconnected_layers]

with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Initialize MediaPipe Hands for gesture recognition
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Function to speak system responses
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize voice commands
def recognize_voice_command():
    with sr.Microphone() as source:
        print("Listening for a command...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10)
            command = recognizer.recognize_google(audio)
            print(f"Command recognized: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that. Try speaking more clearly.")
        except sr.RequestError as e:
            print(f"Error with the recognition service: {e}")
        except sr.WaitTimeoutError:
            print("Listening timeout occurred. No speech detected.")
        return None

# Function to add a new face dynamically
def add_new_face():
    speak("Please provide the image file path for the new face.")
    file_path = input("Enter the path to the image file: ").strip()

    if not os.path.exists(file_path):
        speak("The provided file path does not exist.")
        print("Error: File not found.")
        return

    try:
        new_face_image = face_recognition.load_image_file(file_path)
        new_face_encodings = face_recognition.face_encodings(new_face_image)

        if not new_face_encodings:
            speak("No face detected in the image. Please try again with a clear face image.")
            print("Error: No face detected.")
            return
        
        new_face_encoding = new_face_encodings[0]
        name = input("Enter the name of the person: ").strip()

        if not name:
            speak("Name cannot be empty.")
            print("Error: Name cannot be empty.")
            return

        known_face_encodings.append(new_face_encoding)
        known_face_names.append(name)

        with open("known_faces.pkl", "wb") as f:
            pickle.dump((known_face_encodings[1:], known_face_names[1:]), f)

        speak(f"New face for {name} has been added successfully.")
        print(f"New face added: {name}")

    except Exception as e:
        speak("An error occurred while adding the face. Please try again.")
        print(f"Error: {e}")

# Launch the live camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working.")
    exit()

# Set up MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:
    print("System initialized. Use gestures, face recognition, or voice commands.")
    speak("Voice command system initialized. Press 'v' to issue a command.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive the frame.")
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Face recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # YOLO object detection
        height, width = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        for detection in detections:
            for detect in detection:
                scores = detect[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x, center_y, obj_width, obj_height = (
                        int(detect[0] * width),
                        int(detect[1] * height),
                        int(detect[2] * width),
                        int(detect[3] * height),
                    )
                    x = int(center_x - obj_width / 2)
                    y = int(center_y - obj_height / 2)
                    cv.rectangle(frame, (x, y), (x + obj_width, y + obj_height), (255, 0, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv.imshow("Live Camera", frame)

        if cv.waitKey(1) & 0xFF == ord("v"):
            command = recognize_voice_command()
            if command == "add face":
                add_new_face()
            elif command == "exit":
                speak("Exiting the program.")
                break

cam.release()
cv.destroyAllWindows()
