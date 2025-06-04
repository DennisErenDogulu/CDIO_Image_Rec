#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase
from pybricks.communication import UART
from pybricks.tools import wait

# ───────────────────────── hardware ───────────────────────
ev3 = EV3Brick()
left  = Motor(Port.B)
right = Motor(Port.C)
collector = Motor(Port.A)
robot = DriveBase(left, right, wheel_diameter=55.5, axle_track=104)

uart = UART('serial')          # Bluetooth-SPP

def drive_and_collect(turn_deg, dist_cm):
    ev3.screen.print("TURN", turn_deg, "DIST", dist_cm)
    robot.reset()
    robot.turn(turn_deg)
    robot.straight(dist_cm * 10)      # 1 cm ≈ 10 deg on a standard 55.5 mm wheel
    collector.run_time(500, 800)      # spin intake for 0.8 s
    robot.straight(-30)               # back up 30 cm

# ───────────────────────── main loop ──────────────────────
ev3.speaker.say("Ready")
buf = ""
while True:
    if uart.any():
        buf += uart.read().decode()
        if '\n' in buf:
            line, buf = buf.split('\n', 1)
            try:
                t, d = map(float, line.split(';'))
                drive_and_collect(t, d)
            except ValueError:
                ev3.screen.print("Bad line:", line)
    wait(50)
