import zmq
from multiprocessing import Process
import sys

DATA_SERVER_PATH = "tcp://127.0.0.1:40188"

class DataServer(object):

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(DATA_SERVER_PATH)
        self.store = dict()

    def start(self):
        while True:
            (command, key, value) = self.socket.recv_pyobj()
            if command == 'GET':
                if key in self.store:
                    self.socket.send_pyobj(self.store[key])
                else:
                    self.socket.send_pyobj(None)
            elif command == 'SET':
                self.store[key] = value
                self.socket.send_pyobj(True)
            else:
                raise Exception('Invalid command!')

class DataClient(object):

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(DATA_SERVER_PATH)

    def __getitem__(self, key):
        self.socket.send_pyobj(('GET', key, None))
        return self.socket.recv_pyobj()

    def __setitem__(self, key, value):
        self.socket.send_pyobj(('SET', key, value))
        if not self.socket.recv_pyobj():
            raise Exception('Set failed!')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        DataServer().start()
    else:
        print(DataClient()[sys.argv[1]])
