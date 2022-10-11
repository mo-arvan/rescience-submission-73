import numpy as np
from collections import defaultdict


class Epsilon_MB_Agent:

    def __init__(self,environment, gamma=0.95,epsilon = 0.01):
        
        self.environment=environment
        self.gamma = gamma
        self.epsilon = epsilon
      
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        
        self.ajout_states()

        
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.nSAS[old_state][action][new_state] += 1
                    self.Rsum[old_state][action]+=reward
                    self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]

                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:0.0)   
                    for next_state in self.nSAS[old_state][action].keys():
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    delta=1
                    while delta > 1e-3 :
                        delta=0
                        for visited_state in self.nSA:
                            for taken_action in self.nSA[visited_state]:
                                value_action=self.Q[visited_state][taken_action]
                                self.Q[visited_state][taken_action]=self.R[visited_state][taken_action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[visited_state][taken_action][next_state] for next_state in self.tSAS[visited_state][taken_action]])
                                delta=max(delta,np.abs(value_action-self.Q[visited_state][taken_action]))
                    
    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        
     
        if np.random.random() > (1-self.epsilon) :
            action = np.random.choice(self.environment.actions)
        else:
                q_values = self.Q[state]
                maxValue = max(q_values.values())
                action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    
    def ajout_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.Q[state_1][action]=1/(1-self.gamma)