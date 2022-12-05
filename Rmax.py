import numpy as np
from collections import defaultdict


class Rmax_Agent:

    def __init__(self,environment, gamma=0.95, m=5,Rmax=200,m_uncertain_states=2,correct_prior=True):
        
        
        self.environment=environment
        self.gamma = gamma
        
        self.Rmax=Rmax
        self.m = m
        self.m_uncertain_states=m_uncertain_states
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.counter=self.nSA
        self.step_counter=0
        self.last_model_update=0
        self.max_visits=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.correct_prior=correct_prior
        
        self.ajout_states()
        
        
    def learn(self,old_state,reward,new_state,action):
                    
                    
                    self.nSA[old_state][action] +=1
                    self.nSAS[old_state][action][new_state] += 1
                    self.Rsum[old_state][action]+=reward
                    
                    if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:.0)                      
                    for next_state in self.nSAS[old_state][action]:
                        self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]
                    
                    if self.nSA[old_state][action]>=self.max_visits[old_state][action]: self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
                    else : self.R[old_state][action]=self.Rmax
                    
                    
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
                self.max_visits[state][action]=self.m_uncertain_states
        if not self.correct_prior : self.wrong_prior()
    
    def wrong_prior(self):
        for state in self.environment.states:
            for action in self.environment.actions:
                self.max_visits[state][action]=np.random.randint(self.m,self.m_uncertain_states)
        
