PIN = 26

import RPi.GPIO as GPIO

state = True
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)
while True:
    respond = input("Open or close?")
    state = True if respond=="open" else False
    GPIO_state = GPIO.HIGH if state else GPIO.LOW
    GPIO.output(PIN, GPIO_state)