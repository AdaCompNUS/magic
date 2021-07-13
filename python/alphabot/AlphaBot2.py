import RPi.GPIO as GPIO
import time

class AlphaBot2(object):

    def __init__(self,ain1=12,ain2=13,ena=6,bin1=20,bin2=21,enb=26):
        self.AIN1 = ain1
        self.AIN2 = ain2
        self.BIN1 = bin1
        self.BIN2 = bin2
        self.ENA = ena
        self.ENB = enb
        self.PA  = 20
        self.PB  = 20

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.AIN1,GPIO.OUT)
        GPIO.setup(self.AIN2,GPIO.OUT)
        GPIO.setup(self.BIN1,GPIO.OUT)
        GPIO.setup(self.BIN2,GPIO.OUT)
        GPIO.setup(self.ENA,GPIO.OUT)
        GPIO.setup(self.ENB,GPIO.OUT)
        self.PWMA = GPIO.PWM(self.ENA,500)
        self.PWMB = GPIO.PWM(self.ENB,500)
        self.PWMA.start(self.PA)
        self.PWMB.start(self.PB)
        self.stop()

        def forward(self):
            self.PWMA.ChangeDutyCycle(self.PA)
            self.PWMB.ChangeDutyCycle(self.PB)
            GPIO.output(self.AIN1,GPIO.LOW)
            GPIO.output(self.AIN2,GPIO.HIGH)
            GPIO.output(self.BIN1,GPIO.LOW)
            GPIO.output(self.BIN2,GPIO.HIGH)


        def stop(self):
            self.PWMA.ChangeDutyCycle(0)
            self.PWMB.ChangeDutyCycle(0)
            GPIO.output(self.AIN1,GPIO.LOW)
            GPIO.output(self.AIN2,GPIO.LOW)
            GPIO.output(self.BIN1,GPIO.LOW)
            GPIO.output(self.BIN2,GPIO.LOW)

        def backward(self):
            self.PWMA.ChangeDutyCycle(self.PA)
            self.PWMB.ChangeDutyCycle(self.PB)
            GPIO.output(self.AIN1,GPIO.HIGH)
            GPIO.output(self.AIN2,GPIO.LOW)
            GPIO.output(self.BIN1,GPIO.HIGH)
            GPIO.output(self.BIN2,GPIO.LOW)

        def left(self):
            self.PWMA.ChangeDutyCycle(30)
            self.PWMB.ChangeDutyCycle(30)
            GPIO.output(self.AIN1,GPIO.HIGH)
            GPIO.output(self.AIN2,GPIO.LOW)
            GPIO.output(self.BIN1,GPIO.LOW)
            GPIO.output(self.BIN2,GPIO.HIGH)


        def right(self):
            self.PWMA.ChangeDutyCycle(30)
            self.PWMB.ChangeDutyCycle(30)
            GPIO.output(self.AIN1,GPIO.LOW)
            GPIO.output(self.AIN2,GPIO.HIGH)
            GPIO.output(self.BIN1,GPIO.HIGH)
            GPIO.output(self.BIN2,GPIO.LOW)

        def setPWMA(self,value):
            self.PA = value
            self.PWMA.ChangeDutyCycle(self.PA)

        def setPWMB(self,value):
            self.PB = value
            self.PWMB.ChangeDutyCycle(self.PB)

        def setMotor(self, left, right):
            if((right >= 0) and (right <= 100)):
                GPIO.output(self.AIN1,GPIO.HIGH)
                GPIO.output(self.AIN2,GPIO.LOW)
                self.PWMA.ChangeDutyCycle(right)
            elif((right < 0) and (right >= -100)):
                GPIO.output(self.AIN1,GPIO.LOW)
                GPIO.output(self.AIN2,GPIO.HIGH)
                self.PWMA.ChangeDutyCycle(0 - right)
            if((left >= 0) and (left <= 100)):
                GPIO.output(self.BIN1,GPIO.HIGH)
                GPIO.output(self.BIN2,GPIO.LOW)
                self.PWMB.ChangeDutyCycle(left)
            elif((left < 0) and (left >= -100)):
                GPIO.output(self.BIN1,GPIO.LOW)
                GPIO.output(self.BIN2,GPIO.HIGH)
                self.PWMB.ChangeDutyCycle(0 - left)

if __name__=='__main__':

    Ab = AlphaBot2()
    Ab.forward()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
