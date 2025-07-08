class CircularQueue:
    def __init__(self, size):
        # Initialize the queue with a fixed size
        self.size = size
        self.queue = [None] * size  # Array to hold the queue elements
        self.front = self.rear = 0  # Both pointers start at the front

    def is_empty(self):
        # The queue is empty if front and rear are equal
        return self.front == self.rear

    def is_full(self):
        # The queue is full if rear is one position behind front in a circular manner
        return (self.rear + 1) % self.size == self.front

    def enqueue(self, item):
        # If the queue is full, you cannot enqueue
        if self.is_full():
            print("Queue is full. Cannot enqueue.")
        else:
            self.queue[self.rear] = item  # Place the item at the rear
            self.rear = (self.rear + 1) % self.size  # Update the rear with wrapping around

    def dequeue(self):
        # If the queue is empty, you cannot dequeue
        if self.is_empty():
            print("Queue is empty. Cannot dequeue.")
        else:
            item = self.queue[self.front]  # Get the item at the front
            self.front = (self.front + 1) % self.size  # Update the front with wrapping around
            return item

    def display(self):
        # Print the elements of the queue
        if self.is_empty():
            print("Queue is empty.")
        else:
            idx = self.front
            while idx != self.rear:
                print(self.queue[idx], end=" ")
                idx = (idx + 1) % self.size  # Move circularly to the next element
            print()