import cv2
import mediapipe as mp
import numpy as np

class HandGestureCalculator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.equation = ""
        self.result = None

    def detect_gesture(self, hand_landmarks):
        fingers = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        ]
        
        count = sum(fingers)
        
        if count == 0:
            return "0"
        elif count == 1 and fingers[1]:
            return "1"
        elif count == 2 and fingers[1] and fingers[2]:
            return "2"
        elif count == 3 and fingers[1] and fingers[2] and fingers[3]:
            return "3"
        elif count == 4 and not fingers[0]:
            return "4"
        elif count == 5:
            return "5"
        elif fingers[0] and fingers[1] and not any(fingers[2:]):
            return "+"
        elif fingers[0] and fingers[4] and not any(fingers[1:4]):
            return "*"
        elif fingers[1] and fingers[4] and not any([fingers[0], fingers[2], fingers[3]]):
            return "-"
        elif fingers[0] and fingers[2] and fingers[4] and not fingers[1] and not fingers[3]:
            return "="
        else:
            return None

    def update_equation(self, gesture):
        if gesture in "012345":
            self.equation += gesture
        elif gesture in "+-*" and self.equation and self.equation[-1] not in "+-*":
            self.equation += gesture
        elif gesture == "=" and self.equation:
            self.solve_equation()

    def solve_equation(self):
        try:
            self.result = eval(self.equation)
            self.equation = str(self.result)
        except:
            self.result = "Error"
            self.equation = ""

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image")
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.detect_gesture(hand_landmarks)
                    if gesture:
                        self.update_equation(gesture)

            cv2.putText(image, f"Equation: {self.equation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if self.result is not None:
                cv2.putText(image, f"Result: {self.result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Calculator", image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calculator = HandGestureCalculator()
    calculator.run()