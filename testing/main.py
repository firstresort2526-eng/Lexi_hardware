import RPi.GPIO as GPIO
import time
import requests

BUTTON_PIN = 23
DEBOUNCE_TIME = 1

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

button_state = GPIO.HIGH  # Current stable state
last_stable_time = 0
aud_url = "http://127.0.0.1:8000/button_press"

while True:
    current_reading = GPIO.input(BUTTON_PIN)
    
    if current_reading != button_state:
        current_time = time.time()
        
        if current_time - last_stable_time > DEBOUNCE_TIME:
            button_state = current_reading
            last_stable_time = current_time
            
            if button_state == GPIO.LOW:  # Button pressed
                print("Button pressed!")
                requests.post(url=aud_url)
                
                # Wait for release
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.01)
    
    time.sleep(0.001)  # Small delay to prevent CPU hogging