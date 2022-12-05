import numpy as np
from collections import defaultdict

def count_to_dirichlet(dictionnaire):
    keys,values=[],[]
    for key,value in dictionnaire.items():
        keys.append(key)
        values.append(value)
    results=np.random.dirichlet(values)
    return {keys[i]:results[i] for i in range(len(dictionnaire))}

def cross_validation(nSAS_SA,prior_SA): 
        cv,v=0,[]
        sum_prior=sum(prior_SA.values())
        for next_state,next_state_count in nSAS_SA.items():            
            value=(prior_SA[next_state]-1)/(sum_prior-1)
            cv-=np.log(value)*next_state_count
            v+=[-np.log(value)]*int(next_state_count)
        v=np.array(v)
        cardinal=sum(nSAS_SA.values())
        cross_val =cv/cardinal
        v=(v-cross_val)**2
        variance_cv=np.sum(v)/cardinal
        return cross_val,variance_cv

class basic_Agent:

    def __init__(self,environment, gamma=0.95):
        
        
        self.environment=environment
        self.gamma = gamma
        
        self.R = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.Rsum=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.nSA = defaultdict(lambda: defaultdict(lambda: 0))
        self.nSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
        
        self.tSAS = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))    
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        
        self.step_counter=0
        self.initialize_states()
        
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

                    self.compute_transitions(old_state,new_state,action)
                    self.compute_reward(old_state,reward,action)
                    
                    self.value_iteration()
    
    
    def compute_transitions(self,old_state,new_state,action):
        if self.nSA[old_state][action]==1:self.tSAS[old_state][action]=defaultdict(lambda:0.0)   
        for next_state in self.nSAS[old_state][action].keys():
            self.tSAS[old_state][action][next_state] = self.nSAS[old_state][action][next_state]/self.nSA[old_state][action]

        
    def compute_reward(self,old_state,reward,action):
        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
    
    def initialize_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.Q[state_1][action]=1/(1-self.gamma)
    
                
class Epsilon_MB_Agent(basic_Agent):
    
    def __init__(self, environment, gamma,epsilon):
        self.epsilon = epsilon
        super().__init__(environment,gamma)
        
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

class Rmax_Agent(basic_Agent):
    
    def __init__(self, environment,gamma,Rmax,m,m_uncertain_states,correct_prior):
        self.Rmax=Rmax
        self.m = m
        self.m_uncertain_states=m_uncertain_states
        self.correct_prior=correct_prior
        self.max_visits=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        super().__init__(environment,gamma)
        
    def compute_reward(self,old_state, reward, action):
        if self.nSA[old_state][action]>=self.max_visits[old_state][action]: self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
        else : self.R[old_state][action]=self.Rmax
    
    def initialize_states(self):
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
        if not self.correct_prior : self.use_wrong_prior()
    
    def use_wrong_prior(self):
        for state in self.environment.states:
            for action in self.environment.actions:
                self.max_visits[state][action]=np.random.randint(self.m,self.m_uncertain_states)
    

class BEB_Agent(basic_Agent):
    
    def __init__(self,environment, gamma, beta,coeff_prior,informative,correct_prior):
        self.beta=beta
        self.correct_prior=correct_prior
        self.coeff_prior=coeff_prior
        self.informative=informative
        
        self.bonus=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.prior=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        self.prior_0=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        super().__init__(environment,gamma)
    
    def compute_transitions(self, old_state, new_state, action):
        self.prior[old_state][action][new_state]+=1
        self.prior_0[old_state][action]+=1
        self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
        self.bonus[old_state][action]=self.beta/(1+self.prior_0[old_state][action])
    
    def compute_reward(self, old_state, reward, action):
        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]+self.bonus[old_state][action]
    
    def initialize_states(self):
        self.states=self.environment.states
        for state_1 in self.states:
            for action in self.environment.actions:
                if self.informative :
                    prior_transitions=self.environment.transitions[action][state_1].copy()
                    prior_coeff_transitions={k:self.coeff_prior*1e-5 for k in self.states}
                    for k,v in prior_transitions.items():
                        prior_coeff_transitions[k]=max(self.coeff_prior*v,prior_coeff_transitions[k])
                    self.prior[state_1][action]=prior_coeff_transitions
                    self.prior_0[state_1][action]=sum(prior_coeff_transitions.values())
                else : 
                    for state_2 in self.states : 
                        self.prior[state_1][action][state_2]=self.coeff_prior
                        self.prior_0[state_1][action]=len(self.states)*self.coeff_prior
                self.bonus[state_1][action]=self.beta
                self.Q[state_1][action]=(1+self.beta)/(1-self.gamma)
                for state_2 in self.states:
                    self.tSAS[state_1][action][state_2]=1/len(self.states)
        if not self.correct_prior: self.wrong_prior
    
    def use_wrong_prior(self):
        max_prior=0
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    max_prior=max(max_prior,self.prior[state_1][action][state_2])
        for state_1 in  self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.prior[state_1][action][state_2]=np.random.uniform(self.coeff_prior*1e-5,max_prior)
                self.prior_0[state_1][action]=sum(self.prior[state_1][action].values())
                
class BEBLP_Agent(basic_Agent):
    
    def __init__(self,environment, gamma, beta,step_update,alpha,coeff_prior):
        
        self.beta=beta
        self.step_update=step_update
        self.alpha=alpha
        self.coeff_prior=coeff_prior

        self.bonus=defaultdict(lambda: defaultdict(lambda: 0.0))
        self.prior= defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0.0)))
        
        self.last_k=defaultdict(lambda: defaultdict(lambda: [(0,0)]*self.step_update))
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))
        super().__init__(environment,gamma)
    
    def compute_reward(self, old_state, reward, action):
        self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]+self.bonus[old_state][action]
    
    def compute_transitions(self, old_state, new_state, action):

        self.prior[old_state][action][new_state]+=1
        self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
        
        self.tSAS[old_state][action]=count_to_dirichlet(self.prior[old_state][action])
        self.last_k[old_state][action][self.nSA[old_state][action]%self.step_update]=new_state
        
        if self.nSA[old_state][action]>self.step_update:
            new_dict_nSAS,new_dict_prior={k:v for k,v in self.nSAS[old_state][action].items()},{k:v for k,v in self.prior[old_state][action].items()}       
            for last_seen_state in self.last_k[old_state][action]:
                new_dict_nSAS[last_seen_state]-=1
                new_dict_prior[last_seen_state]-=1
                if new_dict_nSAS[last_seen_state]==0:
                    del new_dict_nSAS[last_seen_state]
            new_CV,new_variance=cross_validation(self.nSAS[old_state][action],self.prior[old_state][action])
            old_CV,old_variance=cross_validation(new_dict_nSAS,new_dict_prior)
            self.LP[old_state][action]=max(old_CV-new_CV+self.alpha*np.sqrt(new_variance),0.001)
            self.bonus[old_state][action]=self.beta/(1+1/np.sqrt(self.LP[old_state][action]))
    
    def initialize_states(self):
        self.states=self.environment.states
        number_states=len(self.states)
        for state_1 in self.states:
            for action in self.environment.actions:
                for state_2 in self.states:
                    self.prior[state_1][action][state_2]=self.coeff_prior
                    self.tSAS[state_1][action][state_2]=1/number_states
                self.Q[state_1][action]=(1+self.beta)/(1-self.gamma)
                self.LP[state_1][action]=np.log(number_states)
                self.bonus[state_1][action]=self.beta/(1+1/np.sqrt(self.LP[state_1][action]))

class RmaxLP_Agent:

    def __init__(self,environment, gamma, Rmax, m, step_update, alpha):
        
        self.Rmax=Rmax
        self.m=m
        self.step_update=step_update
        self.alpha=alpha
        
        self.last_k=defaultdict(lambda: defaultdict(lambda: [(0,0)]*self.step_update))
        self.LP=defaultdict(lambda: defaultdict(lambda: 0.0))
        
        super().__init__(environment,gamma)
        
    def compute_reward(self, old_state, reward, action):
        if self.LP[old_state][action] < self.m :self.R[old_state][action]=self.Rsum[old_state][action]/self.nSA[old_state][action]
        else : self.R[old_state][action]=self.Rmax
    
    



    
        
    