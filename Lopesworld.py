import numpy as np


UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

def choice_dictionary(dictionary):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    chosen_index = [number for number in range(len(keys))]
    chosen_index=np.random.choice(chosen_index,1, p=values)[0]
    return keys[chosen_index]

class Lopes_State():
    def __init__(self,transitions,rewards,transitions_after_change={}):
        self.height = 5
        self.width = 5
        self.grid = np.zeros((self.height,self.width)) 
        self.current_location = (0,0)     
        self.first_location=(0,0)
        
        self.actions = [UP, DOWN, LEFT, RIGHT,STAY]
        self.values=rewards
        self.transitions_after_change=transitions_after_change
        
        self.UP=transitions[UP]
        self.DOWN=transitions[DOWN]
        self.LEFT=transitions[LEFT]
        self.RIGHT=transitions[RIGHT]
        self.STAY=transitions[STAY]
                
        self.states=[(i,j) for i in range(self.height) for j in range(self.width)]
        self.transitions=transitions
        self.uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        self.max_exploration=125
        self.number_steps=0
        self.changed=False
        
    def make_step(self, action):
        self.number_steps+=1
        if self.transitions_after_change!={} and self.number_steps==900:
            number_steps=self.number_steps
            self.__init__(self.transitions_after_change,self.rewards)
            self.changed=True
            self.number_steps=number_steps
        last_location = self.current_location       
        if action == UP:
            self.current_location = choice_dictionary(self.UP[last_location[0]][last_location[1]])        
        elif action == DOWN:
            self.current_location = choice_dictionary(self.DOWN[last_location[0]][last_location[1]])   
        elif action == LEFT:
            self.current_location = choice_dictionary(self.LEFT[last_location[0]][last_location[1]]) 
        elif action == RIGHT:
            self.current_location = choice_dictionary(self.RIGHT[last_location[0]][last_location[1]]) 
        elif action == STAY:
            self.current_location = choice_dictionary(self.STAY[last_location[0]][last_location[1]])
        return self.values[self.current_location], self.current_location
        