class Material:
    def __init__(self, type_):
        self.type = type_


class Product:
    def __init__(self, type_):
        self.type = type_


class Order:
    def __init__(
        self,
        order_id,
        job_type,
        material,
        route,
        process_times,
        arrival_time,
        due_time,
        priority=0,
        precision_req=1.0,
    ):
        self.order_id = int(order_id)
        self.job_type = str(job_type)
        self.material = str(material)
        self.route = list(route)
        self.process_times = list(process_times)
        self.arrival_time = int(arrival_time)
        self.due_time = int(due_time)
        self.priority = int(priority)
        self.precision_req = float(precision_req)
        self.step_idx = 0
        self.finished = False
        # For the "Starter" architecture pipeline:
        #   raw (at X) -> component (after manufacturing at Y) -> done (stored at Z)
        self.stage = "raw"

    def next_process(self):
        if self.finished or self.step_idx >= len(self.route):
            return None
        return self.route[self.step_idx]

    def current_process_time(self, default_time):
        if self.step_idx < len(self.process_times):
            return int(self.process_times[self.step_idx])
        return int(default_time)

    def advance(self):
        self.step_idx += 1
        if self.step_idx >= len(self.route):
            self.finished = True


class Station:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.queue = []

    def add_item(self, item):
        self.queue.append(item)

    def remove_item(self):
        return self.queue.pop(0) if self.queue else None

    def remove_first_product(self):
        for i, item in enumerate(self.queue):
            if isinstance(item, Product):
                return self.queue.pop(i)
        return None
 
    def queue_length(self):
        return len(self.queue)
 
    def clear(self):
        self.queue.clear()
