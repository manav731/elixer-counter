import mss
import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# Allow imports from src
sys.path.append(os.path.dirname(__file__))

from card_classifier import predict_card

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.6
MIN_CONTOUR_AREA = 1500
DEBOUNCE_FRAMES = 4
DETECTION_COOLDOWN = 1.5     # seconds between card detections
ELIXIR_PRINT_INTERVAL = 2.0 # print elixir every 2 seconds

# Elixir rules
START_ELIXIR = 5
MAX_ELIXIR = 10

# Arena region
monitor = {
    "top": 56,
    "left": 1051,
    "width": 919,
    "height": 1318
}

# ---------- RAW (HUMAN) ELIXIR COST TABLE ----------
ELIXIR_COST_RAW = {
    "Archers": 3,
    "Arrows": 3,
    "Knight": 3,
    "Minions": 3,
    "Fireball": 4,
    "Mini P.E.K.K.A": 4,
    "Musketeer": 4,
    "Giant": 5,
    "Spear Goblins": 2,
    "Goblins": 2,
    "Goblin Cage": 4,
    "Goblin Hut": 5,
    "Skeletons": 1,
    "Bomber": 2,
    "Tombstone": 3,
    "Valkyrie": 4,
    "Cannon": 3,
    "Barbarians": 5,
    "Mega Minion": 3,
    "Battle Ram": 4,
    "Electro Spirit": 1,
    "Skeleton Dragons": 4,
    "Fire Spirit": 1,
    "Bomb Tower": 4,
    "Inferno Tower": 5,
    "Bats": 2,
    "Mortar": 4,
    "Flying Machine": 4,
    "Rocket": 6,
    "Goblin Barrel": 3,
    "Guards": 3,
    "Skeleton Army": 3,
    "Baby Dragon": 4,
    "Witch": 5,
    "P.E.K.K.A": 7,
    "Royal Giant": 6,
    "Royal Recruits": 7,
    "Royal Hogs": 5,
    "Dark Prince": 4,
    "Prince": 5,
    "Balloon": 5,
    "Three Musketeers": 9,
    "Ice Spirit": 1,
    "Giant Snowball": 2,
    "Ice Golem": 2,
    "Battle Healer": 4,
    "Giant Skeleton": 6,
    "Goblin Gang": 3,
    "Skeleton Barrel": 3,
    "Dart Goblin": 3,
    "Barbarian Hut": 7,
    "Goblin Giant": 6,
    "Tesla": 4,
    "Elite Barbarians": 6,
    "Furnace": 4,
    "Zappies": 4,
    "Hunter": 4,
    "X-Bow": 6,
    "Golem": 8,
    "Princess": 3,
    "Electro Wizard": 4,
    "Inferno Dragon": 4,
    "Ram Rider": 5,
    "Firecracker": 3,
    "Wall Breakers": 2,
    "Electro Dragon": 5,
    "Ice Wizard": 3,
    "Royal Ghost": 3,
    "Phoenix": 4,
    "Rascals": 5,
    "Heal Spirit": 1,
    "Bowler": 5,
    "Bandit": 3,
    "Magic Archer": 4,
    "Lava Hound": 7,
    "Elixir Golem": 3,
    "Executioner": 5,
    "Lumberjack": 4,
    "Night Witch": 4,
    "Elixir Collector": 6,
    "Cannon Cart": 5,
    "Fisherman": 3,
    "Golden Knight": 4,
    "Skeleton King": 4,
    "Monk": 4,
    "Little Prince": 3
}
# --------------------------------------------------

# -------- NORMALIZATION (CRITICAL FIX) --------
def normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace(".", "")
        .replace("-", " ")
        .replace(" ", "_")
    )

ELIXIR_COST = {
    normalize_name(k): v for k, v in ELIXIR_COST_RAW.items()
}
# --------------------------------------------

def get_elixir_regen_time(match_elapsed):
    if match_elapsed >= 240:
        return 0.9     # triple
    elif match_elapsed >= 120:
        return 1.4     # double
    else:
        return 2.8     # normal

def run_demo():
    sct = mss.mss()
    prev_gray = None
    recent_preds = deque(maxlen=DEBOUNCE_FRAMES)

    # -------- Elixir state --------
    enemy_elixir = START_ELIXIR
    match_start_time = time.time()
    last_elixir_update = match_start_time
    last_elixir_print = match_start_time

    # -------- Detection state --------
    last_detection_time = 0
    last_played_card = None

    cv2.namedWindow("Arena Elixir Tracker", cv2.WINDOW_NORMAL)
    print("Running demo...")
    print("Press 'q' to quit\n")

    while True:
        frame = np.array(sct.grab(monitor))[:, :, :3]
        frame = np.ascontiguousarray(frame)
        now = time.time()

        # -------- Elixir regeneration --------
        match_elapsed = now - match_start_time
        regen_time = get_elixir_regen_time(match_elapsed)

        if enemy_elixir < MAX_ELIXIR:
            elapsed = now - last_elixir_update
            gained = int(elapsed / regen_time)
            if gained > 0:
                enemy_elixir = min(MAX_ELIXIR, enemy_elixir + gained)
                last_elixir_update += gained * regen_time

        # -------- Print elixir every 2 sec --------
        if now - last_elixir_print >= ELIXIR_PRINT_INTERVAL:
            print(f"💧 Enemy elixir: {enemy_elixir}")
            last_elixir_print = now

        # -------- Motion detection --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        prev_gray = gray
        contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

        # -------- Card detection --------
        if contours and (now - last_detection_time) > DETECTION_COOLDOWN:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            crop = frame[y:y+h+pad, x:x+w+pad]

            if crop.size > 0:
                crop = cv2.resize(crop, (128, 128))
                card, conf = predict_card(crop)

                if conf > CONF_THRESHOLD:
                    recent_preds.append(card)

                    if recent_preds.count(card) >= DEBOUNCE_FRAMES:
                        base_cost = ELIXIR_COST.get(card, 0)

                        if base_cost == 0:
                            print(f"⚠️ No elixir cost found for '{card}'")

                        # -------- Mirror logic --------
                        if last_played_card == card:
                            cost = base_cost + 1
                            mirror = " (MIRRORED)"
                        else:
                            cost = base_cost
                            mirror = ""

                        enemy_elixir = max(0, enemy_elixir - cost)
                        last_played_card = card
                        last_detection_time = now
                        recent_preds.clear()

                        print(f"\n🎯 CARD PLAYED: {card}{mirror}")
                        print(f"💧 Elixir used: {cost}")
                        print(f"💧 Enemy elixir now: {enemy_elixir}\n")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Enemy Elixir: {enemy_elixir}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.imshow("Arena Elixir Tracker", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting demo.")
            break

    cv2.destroyAllWindows()
    sct.close()

if __name__ == "__main__":
    run_demo()
