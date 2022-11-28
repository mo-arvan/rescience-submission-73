from Useful_functions import play
from Useful_functions import loading_environments


from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

   
### Experiments ###

environments_parameters=loading_environments()


#First experiment

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2},
            Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'u_m':15,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'coeff_prior':0.001,'alpha':0.4},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2}}

#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]
environments=["Lopes"]
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
nb_iters=1

play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy':0.01,'pas_VI':50}


every_simulation=getting_simulations_to_do(environments,agents,nb_iters)
all_seeds=[i for i in range(len(every_simulation))]


pol_errors,rewards=main_function()
pol,CI_pol, reward, CI_reward=extracting_results(rewards,pol_errors,environments,agents,nb_iters)
save_and_plot(pol,CI_pol,reward,CI_reward,agents,environments,play_params)



