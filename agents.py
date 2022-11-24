import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self,environment, gamma=0.95):
        
        
        self.environment=environment
        self.gamma = gamma
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.step_counter=0
    
        self.ajout_states()
        
        
    def value_iteration(self):
        delta=1
        while delta > 1e-2 :
            delta=0
            for visited_state in self.nSA:
                for taken_action in self.nSA[visited_state]:
                    value_action=self.Q[visited_state][taken_action]
                    self.Q[visited_state][taken_action]=self.R[visited_state][taken_action]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[visited_state][taken_action][next_state] for next_state in self.tSAS[visited_state][taken_action]])
                    delta=max(delta,np.abs(value_action-self.Q[visited_state][taken_action]))
                    
    def choose_action(self):
        self.step_counter+=1
        state=self.environment.current_location
        q_values = self.Q[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.nSAS[old_state][action][new_state] += 1
                    self.Rsum[old_state][action]+=reward
                    
                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:.0)                      
                    for next_state in self.nSAS[old_state][action]:
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    if self.nSA[old_state][action]>=self.max_visits[old_state][action]: self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    else : self.R[old_state][action]=self.Rmax
                    
                    
                    self.value_iteration()
    
    
    def ajout_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                self.R[state_1][action]=self.Rmax
                self.Q[state_1][action]=self.Rmax/(1-self.gamma)
                self.max_visits[state_1][action]=self.m
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/len(self.states)
        for state in self.environment.uncertain_states:
            for action in self.environment.actions:
                self.max_visits[state][action]=self.u_m
        if not self.correct_prior : self.wrong_prior()
    
    def wrong_prior(self):#Below is the wrong prior version
        for state in self.environment.states:
            for action in self.environment.actions:
                self.max_visits[state][action]=np.random.randint(self.m,self.u_m)
                
class Epsilon_MB_Agent(Agent):
    def __init__(self,environment, gamma=0.95,epsilon = 0.01):
        