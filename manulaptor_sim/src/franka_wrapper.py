import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65530  # The port used by the server

class FrankaWrapper: 
    def __init__(self, ip=HOST, port=PORT):
        self.ip = ip
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((ip, port))

    def send(self,data):

        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.connect((self.ip, self.port))
        self.client.sendall(bytes(data,encoding='utf-8'))
        data = self.client.recv(1024)

        print(f"Received {data!r}")

    def get_green_cube(self):
        self.send("green_cube")

    def get_yellow_cube(self):
        self.send("yellow_cube")

    def get_red_cube(self):
        self.send("red_cube")
