class Material:
    def __init__(self, type_):
        self.type = type_


class Product:
    def __init__(self, type_):
        self.type = type_


class Station:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.queue = []

    def add_item(self, item):
        self.queue.append(item)

    def remove_item(self):
        return self.queue.pop(0) if self.queue else None
 
    def queue_length(self):
        return len(self.queue)
 
    def clear(self):
        self.queue.clear()
