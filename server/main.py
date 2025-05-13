import asyncio
import struct
import json
from utils import Camera
import cv2
import base64
import numpy as np
from logger import Logger

pointing_logger = Logger('logs/logs.csv', ['employee', 'date', 'action_type', 'time', 'action'])



cameras = []

def camera_exists(addr):
    for camera in cameras:
        if camera.host == addr[0] and camera.port == addr[1]:
            return True
        
    return False




async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    camera = Camera(addr[0], addr[1], reader, writer)

    while True:
        raw_len = await reader.readexactly(4)
        msg_len = struct.unpack('!I', raw_len)[0]
        data = await reader.readexactly(msg_len)
        message = json.loads(data.decode())

        action_type = message['action_type'] if 'action_type' in message else None

        if action_type == 'pointage':
            pointing_logger.log([
                message['employee'],
                message['date'],
                message['action_type'],
                message['time'],
                message['action']
            ])
        else:
            print("no thing to do")

        image_data = message['image']

        image_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imwrite('test.jpg', image)
        if not data:
            print(f"Connection closed by {addr}")
            break
        message = data.decode()
        if writer:
            writer.write(struct.pack('!I', len("received thank you")))
            writer.write("received thank you".encode('utf-8'))
            await writer.drain()

    writer.close()
    await writer.wait_closed()

async def main():
    server = await asyncio.start_server(
        handle_client, '127.0.0.1', 8765
    )

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()

# لتشغيل السيرفر
asyncio.run(main())
