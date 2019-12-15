import enum
import numpy as np
import serial
from gpiozero import Button

import image_retrieval
from text_detection import *

class Mode(enum.Enum): 
    depth = 1
    objectRecog = 2
    textDetect = 3

# plays an audio file; requires package mpg321 to be installed
def playAudio(filename):
    command = 'mpg321 ' + filename + ' &'
    os.system(command)

def main():
    outfile = "audio/output.mp3"
    
    # depth mode by default
    mode = Mode.depth
    image_retrieval.startDepthMode()

    # set up connections
    nrfSerial = serial.Serial("/dev/ama0", 115200)
    button = Button(4) # TODO: pin number

    while True:
        # read mode change from serial
        line = nrfSerial.readline().strip() #TODO: or should this be on an interrupt

        # read button input from gpio pin
        pressed = button.is_pressed()

        # change mode
        if line in Mode.__members__:
            mode = Mode[line]

        # mode cases
        if mode == Mode.depth:
            # get haptic array, flattened
            arr = image_retrieval.getBinnedDepthArray().flatten()
            # 1D array -> byte string
            arrStr = ' '.join(map(str, arr))
            arrBStr = bytes(arrStr, encoding='utf-8')
            # write to serial
            nrfSerial.write(arrBStr)
            
            # TODO: on leaving this mode, call endDepthMode
        elif mode == Mode.objectRecog:
            if pressed:
                # TODO: put in/call Amy's code
                pass
        elif mode == Mode.textDetect:
            if pressed:
                # get image from realsense cam
                img = image_retrieval.getColorImg()
                
                # photo -> detected text
                text_to_speak = pic_to_text(img)
                # detected text -> synthetic audio
                text_to_speech(text_to_speak, outfile)
                playAudio(outfile)
    
    #TODO: how to turn on/off?

    # cleanup
    if mode == Mode.depth:
        image_retrieval.endDepthMode()


if __name__ == '__main__':
    main()