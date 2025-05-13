import asyncio
from client import Server
from client.camera import Camera

"""
this file is the main class of the client Core Concept
let's go deeper in this concept
1 - the client is camera which has connection to local server through websocket 
        why? to perform real time messaging
        
2 - the camera and the socket connection will run in parallel sharing same memory
"""

class Application:
    def __init__(self):
        self.server = None
        self.camera = None
        self.setup()

        self.shared_data = {}



    def setup(self):
        # first we need to setup the server
        self.server = Server()

        self.camera = Camera()


    async def run(self):
        #await self.server.connect()
        await self.camera.run(self.shared_data, self.server)

