#!/usr/bin/env python3
"""
Start:      python vision_send_xy.py COM6      # ← Windows-porten
Udvikling:  python vision_send_xy.py none      # kun log, ingen send
"""
import sys, os, cv2, numpy as np, math, time, random, logging, threading, serial
from queue import Queue
from typing import List, Tuple
from roboflow import Roboflow

# ---------- CLI -----------------------------------------------------------
PORT = sys.argv[1] if len(sys.argv) > 1 else "none"
def open_ser(p):
    if p.lower() == "none":
        class Dummy:  # viser bare hvad der ville blive sendt
            def write(self, data): print("[DRY]", data.decode().strip())
        return Dummy()
    return serial.Serial(p, 115200, timeout=0.3)
ser = open_ser(PORT)

# ---------- konstanter ----------------------------------------------------
FIELD_W, FIELD_H = 180, 120      # cm
GRID = 10
GOAL_A_H, GOAL_B_H = 8, 20
HSV_TOL = (10,100,100)
BALL_CLASSES = ("whitetabletennisballs","orangetabletennisballs")

# ---------- Roboflow ------------------------------------------------------
rf = Roboflow(api_key="qJTLU5ku2vpBGQUwjBx2")
model = rf.workspace("cdio-nczdp").project("cdio-golfbot2025").version(12).model

# ---------- logging & kamera ---------------------------------------------
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # ← ændr index hvis nødvendigt
if not cap.isOpened(): log.error("kamera?!"); sys.exit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

# ---------- globale tilstande --------------------------------------------
corners: List[Tuple[int,int]] = []
H_px_cm = H_cm_px = None;  ready = False
stage = "corners"          # corners→front→back→track
front_hsv = back_hsv = None
Qin,Qout,stop = Queue(1),Queue(1),threading.Event()
last = None

# ---------- små helpers ---------------------------------------------------
def project(pt):  # cm→px
    v=np.array([*pt,1],np.float32); p=H_px_cm@v; p/=p[2]; return int(p[0]),int(p[1])
def px2cm(pt):
    if not ready: return None
    v=np.array([*pt,1],np.float32); c=H_cm_px@v; c/=c[2]; return float(c[0]),float(c[1])
def hsv_range(c):
    h,s,v=map(int,c); dh,ds,dv=HSV_TOL
    lo=np.array([max(0,h-dh),max(0,s-ds),max(0,v-dv)],np.uint8)
    hi=np.array([min(179,h+dh),min(255,s+ds),min(255,v+dv)],np.uint8)
    return lo,hi

# ---------- mus -----------------------------------------------------------
def mouse(e,x,y,_,__):
    global stage,H_px_cm,H_cm_px,ready,front_hsv,back_hsv
    if e!=cv2.EVENT_LBUTTONDOWN: return
    if stage=="corners":
        corners.append((x,y)); log.info("corner %d %s",len(corners),(x,y))
        if len(corners)==4:
            dst=np.float32([[0,0],[FIELD_W,0],[FIELD_W,FIELD_H],[0,FIELD_H]])
            H_px_cm=cv2.getPerspectiveTransform(dst,np.float32(corners))
            H_cm_px=np.linalg.inv(H_px_cm); ready=True; stage="front"
            log.info("homografi ✔ – klik GRØN lap")
    elif stage=="front":
        hsv=cv2.cvtColor(last,cv2.COLOR_BGR2HSV); front_hsv=hsv[y,x]; stage="back"
        log.info("front=%s – klik BLÅ lap",front_hsv)
    elif stage=="back":
        hsv=cv2.cvtColor(last,cv2.COLOR_BGR2HSV); back_hsv=hsv[y,x]; stage="track"
        log.info("back=%s – ▶ tracking",back_hsv)

# ---------- tråde ---------------------------------------------------------
def grab():
    while not stop.is_set():
        ret,f=cap.read()
        if ret: Qin.put(cv2.resize(f,(416,416)))

def proc():
    global last
    while not stop.is_set():
        if Qin.empty(): time.sleep(0.005); continue
        frame=Qin.get(); last=frame.copy()

        # -- Roboflow
        try: preds=model.predict(frame,confidence=60,overlap=20).json()
        except Exception as e: log.error("RF %s",e); preds={}
        balls=[]
        for p in preds.get('predictions',[]):
            x,y,w,h=map(int,[p['x'],p['y'],p['width'],p['height']])
            lab=p['class']; clean=lab.lower().replace(" ","")
            col=(random.randrange(255),random.randrange(255),random.randrange(255))
            cv2.rectangle(frame,(x-w//2,y-h//2),(x+w//2,y+h//2),col,2)
            cv2.putText(frame,f"{lab}:{p['confidence']:.2f}",(x-w//2,y-h//2-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
            if clean in BALL_CLASSES: balls.append((x,y,lab))

        # -- overlays
        if ready:
            for x in range(0,FIELD_W+1,GRID_STEP):
                cv2.line(frame,project((x,0)),project((x,FIELD_H)),(255,255,255),1)
            for y in range(0,FIELD_H+1,GRID_STEP):
                cv2.line(frame,project((0,y)),project((FIELD_W,y)),(255,255,255),1)
        for c in corners: cv2.circle(frame,c,5,(0,0,255),-1)

        # -- robot-centroid + SEND xy ------------------------
        if stage=="track" and front_hsv is not None and back_hsv is not None and balls:
            hsv=cv2.cvtColor(last,cv2.COLOR_BGR2HSV)
            cf=cv2.moments(cv2.inRange(hsv,*hsv_range(front_hsv)))
            cb=cv2.moments(cv2.inRange(hsv,*hsv_range(back_hsv)))
            if cf["m00"] and cb["m00"]:
                rx,ry=int((cf["m10"]/cf["m00"]+cb["m10"]/cb["m00"])/2),\
                      int((cf["m01"]/cf["m00"]+cb["m01"]/cb["m00"])/2)
                bx,by,_=min(balls,key=lambda b:(b[0]-rx)**2+(b[1]-ry)**2)
                cv2.circle(frame,(bx,by),6,(0,255,255),2)
                ball_cm=px2cm((bx,by))
                if ball_cm:
                    ser.write(f"{ball_cm[0]:.1f};{ball_cm[1]:.1f}\n".encode())
                    log.info("send xy %.1f %.1f",*ball_cm)

        if not Qout.full(): Qout.put(frame)

# ---------- main ----------------------------------------------------------
cv2.namedWindow("Live"); cv2.setMouseCallback("Live",mouse)
threading.Thread(target=grab,daemon=True).start()
threading.Thread(target=proc,daemon=True).start()
while not stop.is_set():
    if not Qout.empty(): cv2.imshow("Live",Qout.get())
    if cv2.waitKey(1)&0xFF in (27,ord('q')): stop.set()
cap.release(); cv2.destroyAllWindows()
