

class Camera:
    def __init__(self, host, port, reader, writer):
        self.host = host
        self.port = port
        self.reader = reader
        self.writer = writer
        self.persons = {}
        print(f"camera init : {host}:{port}")

    def send(self, message):
        self.writer.send(message)