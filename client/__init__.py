import os
from dotenv import load_dotenv
load_dotenv()


def config(key:str, default:str = None):
    return os.getenv(key) or default