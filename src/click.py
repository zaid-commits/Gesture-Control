import cv2
import mediapipe as mp
import numpy as np
import pyautogui

class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.equation = ""
        self.result = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_gesture = None
        self.gesture_cooldown = 0

    def detect_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        fingers_up = [
            thumb_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y,
            index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            middle_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            ring_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            pinky_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        ]

        if sum(fingers_up) == 0:
            return "0"
        elif sum(fingers_up) == 1 and fingers_up[1]:
            return "1"
        elif sum(fingers_up) == 2 and fingers_up[1] and fingers_up[2]:
            return "2"
        elif sum(fingers_up) == 3 and fingers_up[1] and fingers_up[2] and fingers_up[3]:
            return "3"
        elif sum(fingers_up) == 4 and fingers_up[0] == False:
            return "4"
        elif sum(fingers_up) == 5:
            return "5"
        elif fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]):
            return "+"
        elif fingers_up[0] and fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]):
            return "-"
        elif fingers_up[0] and fingers_up[4] and not any(fingers_up[1:4]):
            return "*"
        elif fingers_up[1] and fingers_up[2] and fingers_up[3] and fingers_up[4] and not fingers_up[0]:
            return "/"
        elif fingers_up[0] and fingers_up[4] and fingers_up[2] and not fingers_up[1] and not fingers_up[3]:
            return "="
        else:
            return "unknown"

    def update_equation(self, gesture):
        if gesture in "0123456":
            self.equation += gesture
        elif gesture in "+-*/":
            if self.equation and self.equation[-1] not in "+-*/":
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

    def move_cursor(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        pyautogui.moveTo(x, y)

    def perform_click(self, gesture):
        if gesture == "1" and self.last_gesture != "1":
            pyautogui.click()
        elif gesture == "2" and self.last_gesture != "2":
            pyautogui.rightClick()

    def run(self):
        while True:
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image")
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.detect_gesture(hand_landmarks)
                    
                    if self.gesture_cooldown == 0:
                        self.update_equation(gesture)
                        self.perform_click(gesture)
                        self.gesture_cooldown = 10  # Set cooldown to 10 frames
                    else:
                        self.gesture_cooldown -= 1

                    self.move_cursor(hand_landmarks)
                    self.last_gesture = gesture

            cv2.putText(image, f"Equation: {self.equation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if self.result is not None:
                cv2.putText(image, f"Result: {self.result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Calculator", image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()