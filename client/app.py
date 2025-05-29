import asyncio
from client import Server
from client.camera import Camera

"""
this file is the main class of the client Core Concept
let's go deeper in this concept
1 - the client is camera which has connection to local server through websocket 
        why? to perform real time messaging
        
2 - the camera and the socket connection will run in parallel sharing same memory

the steps of the client working is :
    1 - the camera will load it's state, that's mean it will load its data like known faces, and zones, events
    1.1 - the camera will check if there is a new faces available in the server ( added by admin )
    1.2 - if there is a new faces it will download them, or it will just continue bootstrapping
    1.3 - for fast recognition, the faces will be loaded into run memory ( RAM )
    1.4 - for fast processing, it would be better if the camera integrated with GPU
    1.5 - the camera then will load the model from the memory, we will use multiple models
            like Yolo for person detection, and insightface for face recognition
    2 - if the camera recognize a new person the camera will check if an event is required for that person
            if yes, the event will fired , nothing will happen otherwise
    2.1 - if the camera failed in recognizing a person a message will be sent to the server in realtime
    2.2 - in other cases it would be better if the camera follows the person, for now let's stay fixed
    2.3 - if a recognized person enter to a red zone , the camera will check the authorization for that person
            if not authorized an alert with an image will be sent to the server
    2.4 - if a not recognized person enter to red zone an alert with a frame will be sent to server
    3 - just if the time is with us, implement the re-id in a fast way please.
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
