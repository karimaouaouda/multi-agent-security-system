import abc
from enum import Enum


class MessageType(Enum):
    INFORM = 1
    ALERT = 2
    DANGER = 3
    ERROR = 4
    DEMAND = 5
    ASK = 6
    WAIT = 7
    WARNING = 8

class IOMessage:
    def __init__(self, message_type:MessageType):
        self.type = message_type
        self.binary = None
        self.payload:dict|None = None

    def with_binary(self, binary):
        self.binary = binary
    def append(self, key:str, value:any):
        if self.payload is None:
            self.payload = {}
        self.payload[key] = value


class IO(abc.ABC):
    def __init__(self, ):
        self.message: None|dict = None

    @abc.abstractmethod
    def send(self, message: IOMessage):
        pass

    @abc.abstractmethod
    def recv(self):
        pass