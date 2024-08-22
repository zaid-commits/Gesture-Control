import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

# Start capturing video
cap = cv2.VideoCapture(0)

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
    
    # Display the recognized number
    if number_text:
        cv2.putText(image, number_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    
    # Show the image
    cv2.imshow('Hand Tracking Calculator - Number Recognition', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
