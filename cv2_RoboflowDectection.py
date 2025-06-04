#!/usr/bin/env python3
import cv2, numpy as np, random, threading, time, logging, os
from   queue import Queue
from   typing import List, Tuple
from   roboflow import Roboflow

# ─── bruger-konstanter ────────────────────────────────────────────────────
FIELD_WIDTH_CM, FIELD_HEIGHT_CM = 180, 120
GRID_SPACING_CM                 = 10
GOAL_A_HEIGHT_CM, GOAL_B_HEIGHT_CM = 8, 20

BLOB_TOLER_HSV   = (10, 100, 100)      # ±H,S,V for lap-farver
MASK_MIN_AREA    = 100                 # px² – filtrer støj

# Bold-klasser fra Roboflow (lowercase, uden mellemrum)
BALL_CLASSES = ("whitetabletennisballs", "orangetabletennisballs")

# ─── Roboflow ─────────────────────────────────────────────────────────────
rf    = Roboflow(api_key="qJTLU5ku2vpBGQUwjBx2")
model = rf.workspace("cdio-nczdp").project("cdio-golfbot2025").version(10).model

# ─── OpenCV setup ─────────────────────────────────────────────────────────
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
cv2.setUseOptimized(True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    log.error("Kunne ikke åbne kamera!")
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# ─── globale states ───────────────────────────────────────────────────────
field_corners: List[Tuple[int,int]] = []
H_px_from_cm = None          # cm ➜ pixel
H_cm_from_px = None          # pixel ➜ cm (inverse)
homog_ready  = False

stage        = "corners"     # corners → front → back → track
front_hsv = back_hsv = None

frame_Q = Queue(maxsize=1)
out_Q   = Queue(maxsize=1)
stop    = threading.Event()

# ─── helper: homografi + tegning ──────────────────────────────────────────
def project(pt_cm):
    vec = np.array([pt_cm[0], pt_cm[1], 1.0], np.float32)
    px  = H_px_from_cm @ vec
    px /= px[2]
    return int(px[0]), int(px[1])

def px_to_cm(pt_px):
    if not homog_ready or H_cm_from_px is None:
        return None
    vec = np.array([pt_px[0], pt_px[1], 1.0], np.float32)
    cm  = H_cm_from_px @ vec
    cm /= cm[2]
    return float(cm[0]), float(cm[1])

def draw_field(img):
    for x in range(0, FIELD_WIDTH_CM+1, GRID_SPACING_CM):
        cv2.line(img, project((x,0)), project((x,FIELD_HEIGHT_CM)), (255,255,255),1)
    for y in range(0, FIELD_HEIGHT_CM+1, GRID_SPACING_CM):
        cv2.line(img, project((0,y)), project((FIELD_WIDTH_CM,y)), (255,255,255),1)
    mid = FIELD_HEIGHT_CM/2
    cv2.line(img, project((0,mid-GOAL_A_HEIGHT_CM/2)), project((0,mid+GOAL_A_HEIGHT_CM/2)), (255,0,0),4)
    cv2.line(img, project((FIELD_WIDTH_CM,mid-GOAL_B_HEIGHT_CM/2)),
                  project((FIELD_WIDTH_CM,mid+GOAL_B_HEIGHT_CM/2)), (0,255,0),4)

def draw_clicks(img):
    for p in field_corners:
        cv2.circle(img, p, 5, (0,0,255), -1)

def hsv_range(center):
    h, s, v = map(int, center)      # cast til Python-int først
    dh, ds, dv = BLOB_TOLER_HSV
    low  = np.array([max(0,   h-dh),
                     max(0,   s-ds),
                     max(0,   v-dv)], dtype=np.uint8)
    high = np.array([min(179, h+dh),
                     min(255, s+ds),
                     min(255, v+dv)], dtype=np.uint8)
    return low, high


# ─── mouse callback ───────────────────────────────────────────────────────
def mouse_cb(ev,x,y,flags,param):
    global stage, H_px_from_cm, H_cm_from_px, homog_ready, front_hsv, back_hsv
    if ev != cv2.EVENT_LBUTTONDOWN:
        return

    if stage == "corners":
        field_corners.append((x,y))
        log.info("Corner %d gemt: (%d,%d)", len(field_corners), x, y)
        if len(field_corners) == 4:
            dst = np.float32([[0,0],[FIELD_WIDTH_CM,0],
                              [FIELD_WIDTH_CM,FIELD_HEIGHT_CM],[0,FIELD_HEIGHT_CM]])
            H_px_from_cm = cv2.getPerspectiveTransform(dst, np.float32(field_corners))
            H_cm_from_px = np.linalg.inv(H_px_from_cm)
            homog_ready  = True
            stage = "front"
            log.info("Homografi klar ✓ – klik GRØN front-lap")

    elif stage == "front":
        hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        front_hsv = hsv[y,x]; stage = "back"
        log.info("Front farve = %s – klik BLÅ bag-lap", front_hsv)

    elif stage == "back":
        hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        back_hsv = hsv[y,x]; stage = "track"
        log.info("Bag farve = %s – ▶ tracking starter", back_hsv)

# ─── capture / process tråde ──────────────────────────────────────────────
def grabber():
    while not stop.is_set():
        ret, f = cap.read()
        if ret:
            frame_Q.put(cv2.resize(f, (416,416)))

def processor():
    global last_frame
    while not stop.is_set():
        if frame_Q.empty():
            time.sleep(0.005); continue
        frame = frame_Q.get()
        last_frame = frame.copy()

        # Roboflow
        try:
            preds = model.predict(frame, confidence=60, overlap=20).json()
        except Exception as e:
            log.error("Model fejl: %s", e)
            preds = {}

        for p in preds.get('predictions', []):
            x,y,w,h = map(int, [p['x'],p['y'],p['width'],p['height']])
            label_raw = p['class']
            label_clean = label_raw.lower().replace(" ", "")
            col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

            cv2.rectangle(frame, (x-w//2,y-h//2), (x+w//2,y+h//2), col, 2)
            cv2.putText(frame, f"{label_raw}:{p['confidence']:.2f}",
                        (x-w//2, y-h//2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            if label_clean in BALL_CLASSES:
                pos_cm = px_to_cm((x,y))
                if pos_cm:
                    log.info("📍 %s ved (%.1f cm, %.1f cm)", label_raw, *pos_cm)

        # overlays
        if homog_ready:
            draw_field(frame)
        draw_clicks(frame)

        # robot-tracking (samme som før)
        if stage == "track":
            hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
            fL,fU = hsv_range(front_hsv)
            bL,bU = hsv_range(back_hsv)
            mask_f = cv2.inRange(hsv, fL, fU)
            mask_b = cv2.inRange(hsv, bL, bU)

            def centroid(m):
                M = cv2.moments(m)
                return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) \
                       if M["m00"] > MASK_MIN_AREA else None

            cf = centroid(mask_f); cb = centroid(mask_b)
            if cf and cb:
                cx,cy = (cf[0]+cb[0])//2, (cf[1]+cb[1])//2
                cv2.circle(frame, cf, 4, (0,255,0), -1)
                cv2.circle(frame, cb, 4, (255,0,0), -1)
                cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
                cv2.line(frame, (cx,cy), cf, (0,255,255), 2)

        if not out_Q.full():
            out_Q.put(frame)

# ─── main-loop ────────────────────────────────────────────────────────────
cv2.namedWindow("Live Object Detection")
cv2.setMouseCallback("Live Object Detection", mouse_cb)

threading.Thread(target=grabber, daemon=True).start()
threading.Thread(target=processor, daemon=True).start()

try:
    while not stop.is_set():
        if out_Q.empty():
            time.sleep(0.005); continue
        img = out_Q.get()
        cv2.imshow("Live Object Detection", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            stop.set()
        elif key == ord('r'):
            log.info("Nulstiller kalibrering + farver")
            field_corners.clear()
            homog_ready = False
            stage = "corners"
finally:
    stop.set()
    cap.release()
    cv2.destroyAllWindows()
