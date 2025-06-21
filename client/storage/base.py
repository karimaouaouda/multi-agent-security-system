import abc
import redis

class FileStorage(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get(self, key:str):
        pass

    @abc.abstractmethod
    def set(self, key:str, value:str):
        pass


class RedisStorage(FileStorage):
    def __init__(self, host='localhost', port=6379, db=0):
        super().__init__()
        self._endpoint = redis.Redis(host=host, port=port, db=db)

    def get(self, key:str):
        pass

    def set(self, key:str, value:str):
        pass