#simple PWM control
#Using switch and depth info to control motor
import image2pwm as *
from __future__ import division
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO

switch_add =         #TBD switch gpio pin
#initialize the GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setup(switch_add,GPIO.IN)

#initialize the pwm channel
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

#main loop
while True:
    switch = GPIO.input(switch_add)
    if switch:
        image = read_image
        pwm_signal = depth2pwm(image)
        for i in range(len(pwm_signal)):
            pwm.set_pwm(i,0,pwm_signal[i])
    else:
        for i in range(len(pwm_signal)):
        pwm.set_pwm(i,0,0)
    
    
