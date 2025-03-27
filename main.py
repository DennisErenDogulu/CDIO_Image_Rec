#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
from time import sleep

#! Testkode
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
collection_motor = Motor(Port.A)  
robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track=104)

def collect_ball():
    ev3.speaker.beep()
    collection_motor.run_time(500, 1000)  # Placeholder til at samle bold op, starter motoren i 1 sekund og samler bold op
    sleep(1)


def drop_ball():
    ev3.speaker.beep()
    collection_motor.run_time(-500, 3000)  # Placeholder til at smide bold, starter motoren i 1 sekund og smider bold

def move_pattern():
    for i in range(3):  
        robot.straight(500)  
        collect_ball()
        robot.turn(90)  
        sleep(1)
    ev3.speaker.say("Task complete")
    robot.stop()



    ev3.speaker.say("Returning to drop off balls")
    robot.straight(-500)

    drop_ball()
    ev3.speaker.say("Task complete")
    robot.stop()


move_pattern()
