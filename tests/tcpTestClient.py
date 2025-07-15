import socket

class TCPCommunication():
    def __init__(self):
        self.ip = "100.72.193.15"
        self.port = 5000

        self.buff_size = 1024

    def openServer(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.ip, self.port))
            s.listen(1)

            conn, addr = s.accept()

        self.conn = conn
        self.addr = addr

    def readFromClient(self):
        return self.conn.recv(self.buff_size)

    def connectClient(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.ip, self.port))

    def sendToServer(self, data):
        self.s.sendall(data)

    def closeClientConnection(self):
        self.s.close()

if __name__ == "__main__":
    communicator = TCPCommunication()
    communicator.connectClient()

    sendData = b"this is a test"
    communicator.sendToServer(sendData)