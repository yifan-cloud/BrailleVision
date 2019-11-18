from __future__ import division
import time
import Adafruit_PCA9685

def read_image():
    #read image depth info and return a list of depth value
    #CODE HERE#
    
def depth2pwm(depth_list):
    #transfer the depth info to pwm signal
    max_dep =        #TBD
    min_dep =        #TBD
    max_pwm =        #TBD
    min_pwm =        #TBD
    pwm_signal = []
    def trans(depth):
    #CODE HERE#
        return (depth-min_dep)/(max_dep-min_dep)*(max_pwm-min_pwm) + min_pwm
    for i in depth_list:
       signal = trans(i)
       pwm_signal.append(signal)
    return pwm_signal
    
    
    
    
