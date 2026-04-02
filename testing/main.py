import RPi.GPIO as GPIO
import requests

BUTTON_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN,GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_pressed():
    return GPIO.input(BUTTON_PIN) == GPIO.LOW

aud_url = 'http://0.0.0.0:8000'

while True:
    if button_pressed():
        # Call Aud's button_pres endpoint
        results = requests.post(url=f"{aud_url}/button_press")
        print(results.json()['status'])