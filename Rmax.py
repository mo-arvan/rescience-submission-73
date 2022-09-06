import numpy as np
from collections import defaultdict


class Rmax_Agent:

    def __init__(self,environment, gamma=0.95, m=5,Rmax=200,u_m=2,known_states=True):
        
        self.Rmax=Rmax
        
        self.environment=environment
        self.gamma = gamma
        self.m = m
        self.u_m=u_m
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        self.last_model_update=0
        self.max_visits=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        #self.known_state_action=[]
        if known_states : self.ajout_states()

        
    def learn(self,old_state,reward,new_state,action):
                    
                    self.uncountered_state(new_state)
                    
                    self.nSA[old_state][action] +=1
                    self.nSAS[old_state][action][new_state] += 1
                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:.0)                      
                    for next_state in self.nSAS[old_state][action]:
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    if self.nSA[old_state][action]>=self.max_visits[old_state][action]: self.R[old_state][action]=reward
                    else : self.R[old_state][action]=self.Rmax
                    
                    """if self.nSA[old_state][action] == self.max_visits[old_state][action]:
                        self.last_model_update=self.step_counter
                        self.known_state_action.append((old_state,action))
                        self.tSAS[old_state][action]=defaultdict(lambda:.0)
                        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]                         
                        for next_state in self.nSAS[old_state][action].keys():
                            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                        for j in range(self.VI): #cf formule logarithme Strehl 2009 PAC Analysis
                            for state_known,action_known in self.known_state_action:
                                self.Q[state_known][action_known]=self.R[state_known][action_known]+self.gamma*np.sum([max(self.Q[next_state].values())*self.tSAS[state_known][action_known][next_state] for next_state in self.tSAS[state_known][action_known].keys()])
                    """
                    
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
        self.uncountered_state(state)
        q_values = self.Q[state]
        maxValue = max(q_values.values())
        action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
        return action
    
    def uncountered_state(self,state):
        if state not in self.R.keys():
            for action in self.environment.actions : 
                self.R[state][action]=self.Rmax
                self.tSAS[state][action][state]=1
                self.Q[state][action]=self.Rmax/(1-self.gamma)
                self.max_visits[state][action]=self.m
    
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
                #self.max_visits[state][action]=2**self.environment.entropy[state,action]
        #Below is the wrong prior version
        """for state in self.environment.states:
            for action in self.environment.actions:
                self.max_visits[state][action]=np.random.randint(self.m,self.u_m)"""
        
