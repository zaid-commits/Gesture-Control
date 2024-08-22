import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Function to recognize numbers based on finger positions
def recognize_number(landmarks):
    fingers = []
    # Thumb
    fingers.append(landmarks[4].x > landmarks[3].x)
    # Fingers
    for i in range(8, 21, 4):
        fingers.append(landmarks[i].y < landmarks[i - 2].y)
    
    # Define gestures for numbers
    if fingers == [False, False, False, False, False]:
        return "0"
    elif fingers == [False, True, False, False, False]:
        return "1"
    elif fingers == [False, True, True, False, False]:
        return "2"
    elif fingers == [False, True, True, True, False]:
        return "3"
    elif fingers == [False, True, True, True, True]:
        return "4"
    elif fingers == [True, True, True, True, True]:
        return "5"
    else:
        return None

# Function to move the cursor based on the index finger's position
def move_cursor(landmarks):
    index_finger_tip = landmarks[8]  # Index finger tip landmark
    cursor_x = int(index_finger_tip.x * screen_width)
    cursor_y = int(index_finger_tip.y * screen_height)
    pyautogui.moveTo(cursor_x, cursor_y)

# Function to detect clicking gesture (thumb and index finger touching)
def detect_click(landmarks):
    thumb_tip = landmarks[4]
    index_finger_tip = landmarks[8]
    distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
    return distance < 0.02  # Threshold to detect a click

# Start capturing video
cap = cv2.VideoCapture(0)

clicking = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image and detect hands
    results = hands.process(image)
    
    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    number_text = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize numbers
            number = recognize_number(hand_landmarks.landmark)
            if number is not None:
                number_text = number  # Set the text to display the recognized number
            
            # Move the cursor
            move_cursor(hand_landmarks.landmark)
            
            # Detect click gesture
            if detect_click(hand_landmarks.landmark):
                if not clicking:  # Avoid multiple clicks for one gesture
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False
    
    # Display the recognized number
    if number_text:
        cv2.putText(image, number_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    
    # Show the image
    cv2.imshow('Hand Tracking Cursor Control', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
