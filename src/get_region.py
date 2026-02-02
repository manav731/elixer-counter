import pyautogui
import time

print("Step 1: Move your mouse to the TOP-LEFT corner of the game window.")
time.sleep(4)  

x1, y1 = pyautogui.position()
print(f"Top-left recorded at: ({x1}, {y1})")

print("Step 2: Move your mouse to the BOTTOM-RIGHT corner of the game window.")
time.sleep(4)  

x2, y2 = pyautogui.position()
print(f"Bottom-right recorded at: ({x2}, {y2})")

width = x2 - x1
height = y2 - y1

print("\nUse this monitor region in screen_capture.py:")
print(f'{{"top": {y1}, "left": {x1}, "width": {width}, "height": {height}}}')
