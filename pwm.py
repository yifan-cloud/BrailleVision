from machine import Pin, PWM, UART
import time
   

#Configure the PWM pins.
pwm = []
for i in range(16):
   temp = PWM(i,pin = Pin(i,Pin.OUT),freq = PWM.FREQ_16MHZ,duty = 90,period = 16000)
   pwm.append(temp)
   pwm[i].init()
#Initialize the uart
uart = UART(1,9600)

def control():
   msg = uart.read()
   msg_str = str(msg,'UTF-8')
   print(msg_str)
   depth = [int(i) for i in msg.split()]
   def trans(ori):
      min_vib = 0
      max_vib = 200
      diff_vib = max_vib-min_vib
      diff_depth = 3
      return min_vib + (ori/diff_depth)*diff_vib
   pwm_sig = [trans(i) for i in depth]
   for i in range(16):
      pwm[i].duty(pwm_sig[i])
      
      
while True:
   control()
   time.sleep(5)
   
   

      


