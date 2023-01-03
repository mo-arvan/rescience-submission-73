from main_Functions import evaluate_agents,loading_environments


from agents import Rmax_Agent,BEB_Agent,Epsilon_MB_Agent,EBLP_Agent,RmaxLP_Agent

   
### Experiments ###

environments_parameters=loading_environments()
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'EBLP':EBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'EBLP':EBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}


#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]

### Reproduction of the first figure of Lopes et al. (2012) ###
play_parameters={'trials':100, 'max_step':30, 'screen':0,'photos':[1,2,10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.1},
            Rmax_Agent:{'gamma':0.95, 'm':10,'Rmax':1,'m_uncertain_states':20,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':1,'informative':True},
            EBLP_Agent:{'gamma':0.95,'beta':1.3,'step_update':10,'alpha':0.01,'prior_LP':0.001},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2,'prior_LP':0.04}}

environments=["Lopes"]
nb_iters=1

starting_seed=100
evaluate_agents(environments,agents,nb_iters,play_parameters,agent_parameters,starting_seed)


#Reproduction of the second figure of Lopes et al. (2012)

starting_seed=200

#Reproduction of the third figure of Lopes et al. (2012)
starting_seed=300

#Stronger non-stationarity to find the same third figure as Lopes et al. (2012)
starting_seed=400

#Replication of the first figure of Lopes et al. (2012) with a negative reward of -1
starting_seed=1000
#Replication of the third figure of Lopes et al. (2012) with a negative reward of -1
starting_seed=1500

#Stronger non-stationarity than Lopes et al. (2012) with a negative reward of -1
starting_seed=2000


