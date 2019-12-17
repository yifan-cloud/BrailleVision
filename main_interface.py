import enum
import time
import numpy as np
import os
import serial
import RPi.GPIO as GPIO
import image_retrieval
from computer_vision import *

# pin numbers
QUAD1PIN1 = 2
QUAD1PIN2 = 3
QUAD2PIN1 = 14
QUAD2PIN2 = 15
QUAD3PIN1 = 17
QUAD3PIN2 = 27
QUAD4PIN1 = 23
QUAD4PIN2 = 24
SELECT_BUTTON = 5
ROTARY_LEFT = 6
ROTARY_RIGHT = 13

class Mode(enum.Enum): 
    depth = 0
    objectRecog = 1
    textDetect = 2
mode_lookup = { Mode.depth: 0, Mode.objectRecog: 1, Mode.textDetect: 2 }

# depth mode by default
mode = Mode.depth

outfile1 = 'audio_text_mode.mp3'
outfile2 = 'audio_label_mode.mp3'

def Credential():
    command1 = 'export GOOGLE_APPLICATION_CREDENTIALS=Desktop/texttospeech.json'
    command2 = 'export GOOGLE_APPLICATION_CREDENTIALS=Desktop/liuyujia.json'
    os.system(command1)
    os.system(command2)

# changes mode based on whether changing mode number up or down
def switch_mode(changeIsUp):
    # on dial change    
    old_mode = mode
    new_mode = 0
    if changeIsUp:
        # increase mode num
        new_mode = mode_lookup[mode] + 1 % 3
    else: # down
        # decrease mode num
        new_mode = mode_lookup[mode] + 2 % 3 # +2 instead of -1 to ensure result always positive
    mode = Mode(new_mode)
    
    # handle entering/leaving depth mode
    if old_mode == Mode.depth:
        image_retrieval.endDepthMode()
    elif mode == Mode.depth:
        image_retrieval.startDepthMode()

# plays an audio file; requires package mpg321 to be installed
def playAudio(filename):
    command = 'mpg321 ' + filename + ' &'
    os.system(command)

def main():
    outfile = "audio/output.mp3"
    Credential()

    image_retrieval.startDepthMode()

    # set up GPIO
    GPIO.setmode(GPIO.BOARD) # use physical pin numbers on the GPIO
    GPIO.setup(ROTARY_LEFT, GPIO.IN)
    GPIO.setup(ROTARY_RIGHT, GPIO.IN)
    GPIO.setup(SELECT_BUTTON, GPIO.IN)

    GPIO.output(QUAD1PIN1, GPIO.HIGH)
    GPIO.output(QUAD1PIN2, GPIO.HIGH)
    GPIO.output(QUAD2PIN1, GPIO.HIGH)
    GPIO.output(QUAD2PIN2, GPIO.HIGH)
    GPIO.output(QUAD3PIN1, GPIO.HIGH)
    GPIO.output(QUAD3PIN2, GPIO.HIGH)
    GPIO.output(QUAD4PIN1, GPIO.HIGH)
    GPIO.output(QUAD4PIN2, GPIO.HIGH)

    # set up connections
    #nrfSerial = serial.Serial("/dev/ttyS0", 9600, timeout=0) # no timeout, i.e. don't block on waiting
    # nrfSerial = serial.Serial(
    #     port = '/dev/ttyS0',
    #     baudrate = 9600,
    #     parity = serial.PARITY_NONE,
    #     stopbits = serial.STOPBITS_ONE,
    #     bytesize = serial.EIGHTBITS,
    #     timeout = 1)

    while True:
        # read mode change from serial
        # line = nrfSerial.readline()
        # print(line)

        button_pressed = False

        # parse message from serial if positive int received
        # val_read = line.decode("utf-8").strip()  # byte string -> string stripped of whitespace
        
        # read GPIO input pins
        if GPIO.input(ROTARY_LEFT) == GPIO.HIGH:
            switch_mode(True)
        elif GPIO.input(ROTARY_RIGHT) == GPIO.HIGH:
            switch_mode(False)
        elif GPIO.input(SELECT_BUTTON) == GPIO.HIGH:
            button_pressed = True

        # mode cases
        if mode == Mode.depth:
            # get haptic array
            arr = image_retrieval.getBinnedDepthArray()
            arr = np.fliplr(arr) # flip horizontally
            arr = arr.flatten() # flatten to 1D
            
            # 1D array -> byte string
            # arrStr = ' '.join(map(str, arr))
            # arrBStr = bytes(arrStr, encoding='utf-8')
            # write to serial
            # nrfSerial.write(arrBStr)
        elif mode == Mode.objectRecog:
            if button_pressed:
                imageCap()
                img = image_retrieval.getColorImg()
                with open('whatthe.jpg', 'rb') as image_file:
                    content = image_file.read()
                # photo -> object label
                label = pic_to_label(img)
                print(label)
                text_to_speech(label, outfile2)
                playAudio(outfile2)
        elif mode == Mode.textDetect:
            if button_pressed:
                # get image from realsense cam
                img = image_retrieval.getColorImg()
                
                # photo -> detected text
                text_to_speak = pic_to_text(img)
                # detected text -> synthetic audio
                text_to_speech(text_to_speak, outfile1)
                print(text_to_speak)
                playAudio(outfile1)
    
    # cleanup
    if mode == Mode.depth:
        image_retrieval.endDepthMode()
    GPIO.cleanup()


if __name__ == '__main__':
    main()
