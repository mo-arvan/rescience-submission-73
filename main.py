from Useful_functions import loading_environments,extracting_results,main_function,getting_simulations_to_do,save_and_plot


from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

   
### Experiments ###

environments_parameters=loading_environments()


#Reproduction of the first figure of Lopes et al. (2012)

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2},
            Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'u_m':15,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'coeff_prior':0.001,'alpha':0.4},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2}}

#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]
environments=["Lopes"]
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
nb_iters=3

play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}


every_simulation=getting_simulations_to_do(environments,agents,nb_iters)
all_seeds=[i for i in range(len(every_simulation))]


pol_errors,rewards=main_function(all_seeds,every_simulation,play_params,agent_parameters)
pol,CI_pol, reward, CI_reward=extracting_results(rewards,pol_errors,environments,agents,nb_iters)
save_and_plot(pol,CI_pol,reward,CI_reward,agents,environments,play_params,environments,agent_parameters)

#Reproduction of the second figure of Lopes et al. (2012)


#Reproduction of the third figure of Lopes et al. (2012)

#Stronger non-stationarity to find the same third figure as Lopes et al. (2012)

#Replication of the first figure of Lopes et al. (2012) with a negative reward of -1

#Replication of the third figure of Lopes et al. (2012) with a negative reward of -1

#Stronger non-stationarity than Lopes et al. (2012) with a negative reward of -1



