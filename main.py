import numpy as np
import matplotlib.pyplot as plt
import time

from Useful_functions import play
from Useful_functions import plot3D,convergence,save_pickle, open_pickle,value_iteration,policy_evaluation, loading_environments


from Lopesworld import Lopes_State



from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

#Initializing parameters



def loop_play(play_parameters,names_environments=['Lopes'],agents_tested={'Epsilon_MB':Epsilon_MB_Agent},number_of_iterations=1):
    rewards,pol_error={},{}
    for name_environment in names_environments:   
        for name_agent,agent in agents_tested.items(): 
            for iteration in range(number_of_iterations):
                environment=Lopes_State(**environments_parameters[name_environment])
            
                globals()[name_agent]=agent(environment,**agent_parameters[agent]) #Defining a new agent from the dictionary agents
            
                reward,policy_value_error= play(environment,globals()[name_agent],**play_parameters) #Playing in environment
            
                rewards[(name_agent,name_environment,iteration)]=reward
                pol_error[(name_agent,name_environment,iteration)]=policy_value_error
    
    global time_end
    time_end=str(round(time.time()))
    results={'nb_iters':number_of_iterations,'play_parameters':play_parameters,'agent_parameters':agent_parameters,'agents':agents_tested,'environments':names_environments,'rewards':rewards,'pol_error':pol_error}
    np.save('Results/'+time_end+'_polerror.npy',pol_error)
    save_pickle(results,'Results/'+time_end+'.pickle')
    return rewards, pol_error



### Extracting results ###

def extracting_results(rewards,pol_error,names_environments,agents_tested,number_of_iterations):
    mean_pol_error_agent={name_agent: np.average([np.average([pol_error[name_agent,name_environment,i] for i in range(number_of_iterations)],axis=0) for name_environment in names_environments],axis=0) for name_agent in agents_tested.keys()}
    std_pol_error_agent={name_agent:np.std([pol_error[name_agent,name_environment,i] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) for name_agent in agents_tested.keys()}
    rewards_agent={name_agent: np.average([np.average([rewards[name_agent,name_environment,i] for i in range(number_of_iterations)],axis=0) for name_environment in names_environments],axis=0) for name_agent in agents_tested.keys()}
    std_rewards_agent={name_agent: np.std(np.array([rewards[name_agent,name_environment,iteration] for name_environment in names_environments for iteration in range(number_of_iterations)]),axis=0)[0] for name_agent in agents_tested.keys()}
    return mean_pol_error_agent,std_pol_error_agent, rewards_agent, std_rewards_agent

###Basic visualisation ###
    
def plot(pol,std_pol,reward,std_reward,agents_tested,names_environments,play_parameters):
    rename={'RA':'R-max','BEB':'BEB','BEBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'Ɛ-greedy'}
    colors={'RA':'royalblue','RALP':'royalblue','Epsilon_MB':'red','BEB':'black','BEBLP':'black'}
    linewidths={'RA':'0.75','RALP':'1.25','BEB':'0.75','BEBLP':'1.25','Epsilon_MB':'0.75'}
    marker_sizes={'RA':'3','RALP':'3','BEB':'3','BEBLP':'3','Epsilon_MB':'3'}
    
    markers={'RA':'^','RALP':'o','BEB':'x','BEBLP':'*','Epsilon_MB':'s'}
    fig=plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for name_agent in agents_tested.keys():
        plt.errorbar([play_parameters['pas_VI']*i for i in range(len(pol[name_agent]))],pol[name_agent], 
                     yerr=std_pol[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
                     elinewidth=0.5,label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent],fillstyle='none',capsize=1)
    plt.xlabel("Steps")
    plt.ylabel("Policy value error")
    plt.grid(linestyle='--')
    plt.ylim([-12.5,0.5])
    plt.legend()
    plt.savefig('Results/pol_error'+time_end+names_environments[0]+'.png')
    plt.show()

    fig_reward=plt.figure(dpi=300)
    ax_reward=fig_reward.add_subplot(1,1,1)
    for name_agent in agents_tested.keys():
        plt.errorbar([i+1 for i in range(play_parameters['trials'])],reward[name_agent], 
                     yerr=std_reward[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
                     elinewidth=0.5,label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent],fillstyle='none',capsize=1)
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Results/Rewards'+time_end+names_environments[0]+'.png')
    plt.show()
    
###Actual experiment ###

environments_parameters=loading_environments()

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2},
            Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'u_m':15,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'coeff_prior':0.001,'alpha':0.4},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2}}

#First experiment
np.random.seed(10)

environments=['Lopes']
agents={'Epsilon_MB':Epsilon_MB_Agent}
nb_iters=5

play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,100,199,300,499],'accuracy':0.01,'pas_VI':50}

rewards,pol_error=loop_play(play_params,environments,agents,nb_iters)
pol,std_pol, reward, std_reward=extracting_results(rewards,pol_error,environments,agents,nb_iters)
plot(pol,std_pol,reward,std_reward,agents,environments,play_params)





