import numpy as np
import matplotlib.pyplot as plt
import time

from Useful_functions import play
from Useful_functions import save_pickle, loading_environments


from Lopesworld import Lopes_State



from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent



#Main function 


from multiprocessing import Pool


def fast_loop(args):
    return play_with_params(args[1][0],args[1][1],args[1][2],args[1][3],args[0],args[2])


def main_function() :
    before=time.time()
    all_args=[[play_params,every_simulation[seed],all_seeds[seed]] for seed in range(len(all_seeds))]
    pool = Pool()
    results=pool.map(fast_loop,all_args)
    pool.close()
    pool.join()
    pol_errors,rewards={},{}
    for result in results : 
        pol_errors[result[0]]=result[1][1]
        rewards[result[0]]=result[1][0]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    return pol_errors,rewards


### Extracting results ###

def extracting_results(rewards,pol_error,names_environments,agents_tested,number_of_iterations):
    mean_pol_error_agent={name_agent: np.average([np.average([pol_error[name_agent,name_environment,i] for i in range(number_of_iterations)],axis=0) for name_environment in names_environments],axis=0) for name_agent in agents_tested}
    CI_pol_error_agent={name_agent:1.96*np.std([pol_error[name_agent,name_environment,i] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) for name_agent in agents_tested}
    rewards_agent={name_agent: np.average([np.average([rewards[name_agent,name_environment,i] for i in range(number_of_iterations)],axis=0) for name_environment in names_environments],axis=0) for name_agent in agents_tested}
    CI_rewards_agent={name_agent:1.96*np.std([rewards[name_agent,name_environment,i] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) for name_agent in agents_tested}
    return mean_pol_error_agent,CI_pol_error_agent, rewards_agent, CI_rewards_agent

###Basic visualisation ###
    
def save_and_plot(pol,CI_pol,reward,CI_reward,agents_tested,names_environments,play_parameters):
    time_end=str(round(time.time()%1e7))
    results={'play_parameters':play_parameters,'agent_parameters':agent_parameters,'agents':agents_tested,'environments':names_environments,'rewards':reward,'pol_error':pol}
    np.save('Results/'+time_end+'_polerror.npy',pol)
    save_pickle(results,'Results/'+time_end+'.pickle')
    
    rename={'RA':'R-max','BEB':'BEB','BEBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'Ɛ-greedy'}
    colors={'RA':'#9d02d7','RALP':'#0000ff','Epsilon_MB':"#ff7763",'BEB':"#ffac1e",'BEBLP':"#009435"}
    linewidths={'RA':'0.75','RALP':'1.25','BEB':'0.75','BEBLP':'1.25','Epsilon_MB':'0.75'}
    marker_sizes={'RA':'3','RALP':'3','BEB':'3','BEBLP':'3','Epsilon_MB':'3'}
    
    markers={'RA':'^','RALP':'o','BEB':'x','BEBLP':'*','Epsilon_MB':'s'}
    fig=plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for name_agent in agents_tested.keys():
        yerr0 = pol[name_agent] - CI_pol[name_agent]
        yerr1 = pol[name_agent] + CI_pol[name_agent]

        plt.fill_between([play_parameters['pas_VI']*i for i in range(len(pol[name_agent]))], yerr0, yerr1, color=colors[name_agent], alpha=0.2)

        plt.plot([play_parameters['pas_VI']*i for i in range(len(pol[name_agent]))],pol[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
                     label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent])
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
        yerr0 = reward[name_agent] - CI_reward[name_agent]
        yerr1 = reward[name_agent] + CI_reward[name_agent]

        plt.fill_between([i+1 for i in range(play_parameters['trials'])], yerr0, yerr1, color=colors[name_agent], alpha=0.2)
        
        plt.plot([i+1 for i in range(play_parameters['trials'])],reward[name_agent], 
                     color=colors[name_agent],linewidth=linewidths[name_agent],
                     label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent])
    plt.xlabel("Trials")
    plt.ylabel("Reward")
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Results/Rewards'+time_end+names_environments[0]+'.png')
    plt.show()
 
    
def play_with_params(name_environment,name_agent,agent,iteration,play_parameters,seed):
    np.random.seed(seed)
    environment=Lopes_State(**environments_parameters[name_environment])
    globals()[name_agent]=agent(environment,**agent_parameters[agent])
    return (name_agent,name_environment,iteration),play(environment,globals()[name_agent],**play_parameters)
    
def getting_simulations_to_do(names_environments=['Lopes'],agents_tested={'Epsilon_MB':Epsilon_MB_Agent},number_of_iterations=2):
    every_simulation=[]
    for name_environment in names_environments:   
        for name_agent,agent in agents_tested.items(): 
            for iteration in range(number_of_iterations):
                    every_simulation.append((name_environment,name_agent,agent,iteration))
    return every_simulation  
    
   
### Experiments ###

environments_parameters=loading_environments()

agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2},
            Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'u_m':15,'correct_prior':True},
            BEB_Agent:{'gamma':0.95,'beta':3,'correct_prior':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'coeff_prior':0.001,'alpha':0.4},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.3,'m':2}}

#First experiment
#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]
environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,6) for non_stat in range(1,5)]
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
nb_iters=1

play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy':0.01,'pas_VI':50}


every_simulation=getting_simulations_to_do(environments,agents,nb_iters)
all_seeds=[i for i in range(len(every_simulation))]


pol_errors,rewards=main_function()
pol,CI_pol, reward, CI_reward=extracting_results(rewards,pol_errors,environments,agents,nb_iters)
save_and_plot(pol,CI_pol,reward,CI_reward,agents,environments,play_params)



