# Welcome to part 2 - Projector

The code is meant to be run on a raspberry pi 5, NOT ON YOUR COMPUTER.
You should connect a ST77916 LCD to your RPi before running it.
The connection are as follows:

- SI0 = 10 # GPIO 10 - Physical pin 19
- SI1 = 22 # GPIO 9 - Physical pin 21
- SI2 = 27 # GPIO 11 - Physical pin 23
- SI3 = 17 # GPIO 8 - Physical pin 24
- SCL = 11 # GPIO 7 - Physical pin 26
- CS0 = 5 # GPIO 5 - Physical pin 29
- RST = 25 # GPIO 25 - Physical pin 22
- BL = 16 # GPIO 16 - Physical pin 36

The code writes raw bytes to the LCD. Therefore, very little libraries are needed.

1. RPi.GPIO
2. time
3. json
4. numpy
5. pathlib

As these are all installed in the root of RPi, I didn't create requirements.txt
