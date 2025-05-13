import abc

class Event(abc.ABC):
    def __init__(self, server):
        self.server:Server = server

    def send_to_server(self):
        self.server.send(
            self.to_dict()
        )

    @abc.abstractmethod
    def to_dict(self):
        pass

class PersonEntryEvent(Event):
    def __init__(self, server, person):
        super().__init__(server)
        self.person = person

    def to_dict(self):
        return {
            'person': self.person.name,
        }

class PersonExitEvent(Event):
    def __init__(self, server, person):
        super().__init__(server)
        self.person = person

    def to_dict(self):
        return {
            'person': self.person.name,
        }

class EnterToRedPlaceEvent(Event):
    def __init__(self, server, person, place):
        super().__init__(server)
        self.person = person
        self.place = place

    def to_dict(self):
        return {
            'person': self.person.name,
            'place': self.place.name,
        }

class EnterToWhitePlaceEvent(Event):
    def __init__(self, server, person, place):
        super().__init__(server)
        self.person = person
        self.place = place

    def to_dict(self):
        return {
            'person': self.person.name,
            'place': self.place.name,
        }
    


