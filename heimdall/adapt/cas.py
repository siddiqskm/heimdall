# adapt/cas.py

class CAS:
    def __init__(self):
        self.turns = 0
        self.active = True

    def step(self):
        self.turns += 1
        if self.turns > 15:
            self.active = False
