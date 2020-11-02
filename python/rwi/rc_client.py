import zmq
import time
import keyboard

PWM_HIGH = 20
PWM_MID = 10

class RCClient():

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://10.42.0.10:41273")

    def send(self, left, right):
        self.socket.send_json((left, right))
        return self.socket.recv_json() == 0

    def ping(self):
        self.socket.send_json("ping")
        return self.socket.recv_json() == 0

    def close(self):
        self.send(0, 0)
        self.socket.term()


if __name__ == '__main__':

    client = RCClient()
    while True:

        if keyboard.is_pressed('up'):
            if keyboard.is_pressed('left'):
                client.send(PWM_MID, PWM_HIGH)
            elif keyboard.is_pressed('right'):
                client.send(PWM_HIGH, PWM_MID)
            else:
                client.send(PWM_HIGH, PWM_HIGH)
        elif keyboard.is_pressed('down'):
            if keyboard.is_pressed('left'):
                client.send(-PWM_MID, -PWM_HIGH)
            elif keyboard.is_pressed('right'):
                client.send(-PWM_HIGH, -PWM_MID)
            else:
                client.send(-PWM_HIGH, -PWM_HIGH)
        elif keyboard.is_pressed('left'):
            client.send(-PWM_HIGH, PWM_HIGH)
        elif keyboard.is_pressed('right'):
            client.send(PWM_HIGH, -PWM_HIGH)
        else:
            client.send(0, 0)
