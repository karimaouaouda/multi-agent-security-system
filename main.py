from client.app import Application
import asyncio

if __name__ == '__main__':
    client = Application()
    asyncio.run(client.run())

