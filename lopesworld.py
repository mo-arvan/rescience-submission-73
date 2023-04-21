import numpy as np


UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


class Lopes_environment():

    def __init__(self, transitions, rewards, transitions_after_change=[]):
        self.first_location = 0
        self.current_location = 0

        self.actions = [UP, DOWN, LEFT, RIGHT, STAY]
        self.transitions = transitions
        self.transitions_after_change = transitions_after_change
        self.rewards = rewards

        self.states = np.arange(25)
        self.uncertain_states = [1, 3, 11, 13]
        self.total_steps = 0
        self.changed = False

    def make_step(self, action):
        self.total_steps += 1

        if self.transitions_after_change != [] and self.total_steps == 900:
            total_steps = self.total_steps
            self.__init__(self.transitions_after_change, self.rewards)
            self.changed = True
            self.total_steps = total_steps
        probabilities = self.transitions[self.current_location][action]
        self.current_location = np.random.choice(np.arange(25), size=1, p=probabilities)[0]

        return self.rewards[self.current_location], self.current_location
