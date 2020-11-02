import RPi.GPIO as GPIO
from AlphaBot2 import AlphaBot2
import time
import zmq

Ab = AlphaBot2()

BUZ = 4
def beep_on():
    GPIO.output(BUZ, GPIO.HIGH)
def beep_off():
    GPIO.output(BUZ, GPIO.LOW)

if __name__ == '__main__':
    try:
        # Initialize devices.
        GPIO.setup(BUZ,GPIO.OUT) 

        # Initialize zmq server.
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://10.42.0.10:41273")

        # Alert ready.
        #for _ in range(2):
        #    beep_on()
        #    time.sleep(0.1)
        #    beep_off()
        #    time.sleep(0.05)

        print('Ready.')

        # RC loop.
        while True:
            msg = socket.recv_json()
            if type(msg) is tuple or type(msg) is list:
                (left, right) = msg
                Ab.setMotor(-right, -left) # Flip since forward is in diff direction.
                socket.send_json(0)
                continue
            elif type(msg) is str:
                if msg == "ping":
                    beep_on()
                    time.sleep(0.1)
                    beep_off()
                    time.sleep(0.05)
                    socket.send_json(0)
                    continue
            socket.send_json(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
