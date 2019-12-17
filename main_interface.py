import enum
import time
import numpy as np
import os
import serial
from gpiozero import Button
import RPi.GPIO
import image_retrieval
from computer_vision import *

class Mode(enum.Enum): 
    depth = 0
    objectRecog = 1
    textDetect = 2

outfile1 = 'audio_text_mode.mp3'
outfile2 = 'audio_label_mode.mp3'

def Credential():
    command1 = 'export GOOGLE_APPLICATION_CREDENTIALS=Desktop/texttospeech.json'
    command2 = 'export GOOGLE_APPLICATION_CREDENTIALS=Desktop/liuyujia.json'
    os.system(command1)
    os.system(command2)

# plays an audio file; requires package mpg321 to be installed
def playAudio(filename):
    command = 'mpg321 ' + filename + ' &'
    os.system(command)

def main():
    outfile = "audio/output.mp3"
    Credential()
    # depth mode by default
    mode = Mode.depth
    image_retrieval.startDepthMode() 

    # set up connections
    #nrfSerial = serial.Serial("/dev/ttyS0", 9600, timeout=0) # no timeout, i.e. don't block on waiting
    nrfSerial = serial.Serial(
        port = '/dev/ttyS0',
        baudrate = 9600,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,
        timeout = 1)
    
    button = Button(4) # TODO: pin number

    while True:
        # read mode change from serial
        line = nrfSerial.readline()
        print(line)

        # read button input from gpio pin
        #pressed = button.is_pressed()

        # change mode if positive int received
        old_mode = mode
        val_read = line.decode("utf-8").strip()  # byte string -> string stripped of whitespace
        if val_read.isdigit():
            mode = Mode( int(val_read) ) # string -> int -> Mode

            # handle entering/leaving depth mode
            if old_mode == Mode.depth:
                image_retrieval.endDepthMode()
            elif mode == Mode.depth:
                image_retrieval.startDepthMode()

        # mode cases
        if mode == Mode.depth:
            # get haptic array
            arr = image_retrieval.getBinnedDepthArray()
            arr = np.fliplr(arr) # flip horizontally
            arr = arr.flatten() # flatten to 1D
            # 1D array -> byte string
            arrStr = ' '.join(map(str, arr))
            arrBStr = bytes(arrStr, encoding='utf-8')
            # write to serial
            nrfSerial.write(arrBStr)
        elif mode == Mode.objectRecog:
            if pressed:
                imageCap()
                img = image_retrieval.getColorImg()
                with open('whatthe.jpg', 'rb') as image_file:
                    content = image_file.read()
                # photo -> detected text
                label = pic_to_label(img)
                print(label)
                text_to_speech(label, outfile2)
                playAudio(outfile2)
                pass
        elif mode == Mode.textDetect:
            if pressed:
                # get image from realsense cam
                img = image_retrieval.getColorImg()
                
                # photo -> detected text
                text_to_speak = pic_to_text(img)
                # detected text -> synthetic audio
                text_to_speech(text_to_speak, outfile1)
                print(text_to_speak)
                playAudio(outfile1)
    
    #TODO: how to turn on/off?

    # cleanup
    if mode == Mode.depth:
        image_retrieval.endDepthMode()


if __name__ == '__main__':
    main()
