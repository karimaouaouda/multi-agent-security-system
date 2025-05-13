import asyncio
import struct
import json

class Server:
    def __init__(self, ip = '127.0.0.1', socket_port = 8765):
        self.ip = ip
        self.socket_port = socket_port
        self.ws_reader = None
        self.ws_writer = None

    async def connect(self):
        self.ws_reader, self.ws_writer = await asyncio.open_connection(self.ip, self.socket_port)


    async def listen(self):
        while True:
            raw_len = await self.ws_reader.readexactly(4)
            msg_len = struct.unpack('!I', raw_len)[0]
            data = await self.ws_reader.readexactly(msg_len)
            try:
                message = json.loads(data.decode())
                print("Received:", data.decode().strip())
            except json.decoder.JSONDecodeError:
                print("Received invalid JSON")
            if not data:
                break


    async def start(self, shared_state):
        while True:
            message = input("You: ")
            message = f"{message}\n"
            self.ws_writer.write(message.encode())
            await self.ws_writer.drain()

    async def ping(self, data):
        self.ws_writer.write(data)
        await self.ws_writer.drain()