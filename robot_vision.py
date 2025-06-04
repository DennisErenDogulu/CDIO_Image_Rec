#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.communication import UART
from pybricks.robotics import DriveBase
import math

WHEEL = 55.5   # mm
AXLE  = 104    # mm

ev3 = EV3Brick()
drv = DriveBase(Motor(Port.B), Motor(Port.C), wheel_diameter=WHEEL, axle_track=AXLE)
uart = UART('serial')
ev3.speaker.say("XY ready")

def go_to(x_cm, y_cm):
    # antag robot starter på (0,0) vendt mod +X (juster selv efter behov)
    dx, dy = x_cm, y_cm
    dist = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))
    drv.turn(angle)            # drej
    drv.straight(dist*10)      # kør (cm→deg)
    drv.straight(-30)          # ryk fri

buf=""
while True:
    if uart.any():
        buf += uart.read().decode()
        if '\n' in buf:
            ln, buf = buf.split('\n',1)
            try:
                x,y = map(float, ln.split(';'))
                go_to(x, y)
            except ValueError:
                ev3.screen.print("bad:", ln)
    wait(50)