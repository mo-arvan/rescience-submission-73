from Useful_functions import evaluate_agents,loading_environments


from agents import Rmax_Agent
from agents import BEB_Agent
from agents import Epsilon_MB_Agent
from agents import BEBLP_Agent
from agents import RmaxLP_Agent

   
### Experiments ###

environments_parameters=loading_environments()
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'RALP':RmaxLP_Agent}

#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]

### Reproduction of the first figure of Lopes et al. (2012) ###
play_parameters={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2},
            Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'m_uncertain_states':15,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':1,'step_update':10,'alpha':0.4,'prior_LP':0.001,},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2,'prior_LP':0.001}}

environments=["Lopes"]
nb_iters=20

starting_seed=0
evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)


#Reproduction of the second figure of Lopes et al. (2012)


#Reproduction of the third figure of Lopes et al. (2012)

#Stronger non-stationarity to find the same third figure as Lopes et al. (2012)

#Replication of the first figure of Lopes et al. (2012) with a negative reward of -1

#Replication of the third figure of Lopes et al. (2012) with a negative reward of -1

#Stronger non-stationarity than Lopes et al. (2012) with a negative reward of -1



