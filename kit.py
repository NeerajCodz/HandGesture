import numpy as np
import pyautogui as pg
import mediapipe as mp

pg.PAUSE = 0
pg.MINIMUM_DURATION = 0
pg.MINIMUM_SLEEP = 0

def distance(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5

def LineMag(points: list):
    total_distance = 0.0
    for i in range(len(points) - 1):
        total_distance += distance(points[i], points[i + 1])
    return total_distance

def R_Sq(points):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return r_squared

def slow_scroll(px):
    if px > 0:
        for i in range(int(px / 10)):
            pg.scroll(10)
            pg.sleep(0.004)
    else:
        for i in range(int(px / 10) * -1):
            pg.scroll(-10)
            pg.sleep(0.004)

def get_swipe_direction(path):
    print ("LEN :",len(path))
    if len(path) < 5:
        return None  # Insufficient points to determine direction
    num_fingers=3
    start_x, start_y = path[0]
    end_x, end_y = path[-1]

    delta_x = end_x - start_x
    delta_y = end_y - start_y

    min_distance_threshold = 100

    if num_fingers == 2:
        if abs(delta_x) > abs(delta_y):
            if abs(delta_x) > min_distance_threshold:
                if delta_x > 0:
                    pg.scroll(1)
                    return "Right"
                else:
                    pg.scroll(-1)
                    return "Left"
        else:
            if abs(delta_y) > min_distance_threshold:
                if delta_y > 0:
                    pg.scroll(-1)
                    return "Up"
                else:
                    pg.scroll(1)
                    return "Down"
    elif num_fingers == 3:
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                pg.hotkey("alt","tab")
                return "Right Window"
            else:
                pg.hotkey("alt","tab","left", "left")
                return "Left Window"
        else:
            if delta_y > 0:
                pg.hotkey("winleft", "down")
                return "Minimize"
            else:
                pg.hotkey("winleft", "up")
                return "Maximize"
    elif num_fingers == 4:
        if delta_x > 0:
            pg.hotkey("winleft", "ctrl", "right")
            return "Right Desktop"
        else:
            pg.hotkey("winleft", "ctrl", "left")
            return "Left Desktop"
    elif num_fingers == 5:
        if abs(delta_y) > min_distance_threshold:
            if delta_y > 0:
                slow_scroll(-700)
                return "Minimize"
            else:
                slow_scroll(700)
                return "Finger Scroll Up"

    return None

def is_fist_gesture(hand_landmarks):
    if hand_landmarks is None:
        return False

    # Get landmarks for the thumb, index, middle, ring, and pinky fingers
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # Placeholder logic: Check if the fingers are in proximity (e.g., check Y-coordinate)
    # You may need to adjust the thresholds based on your specific hand tracking data
    proximity_threshold = 0.05

    if thumb_tip.y > index_tip.y - proximity_threshold and \
            thumb_tip.y > middle_tip.y - proximity_threshold and \
            thumb_tip.y > ring_tip.y - proximity_threshold and \
            thumb_tip.y > pinky_tip.y - proximity_threshold:
        return True

    return False
