import numpy as np
import matplotlib.pyplot as plt
import time
import random

from Useful_functions import play
from Useful_functions import plot3D,convergence,save_pickle, open_pickle,value_iteration,policy_evaluation


from Lopesworld import Lopes_State
from Lopes_no_stat import Lopes_nostat


from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent

#Initializing parameters
environments_parameters={}
all_environments={}
for number_world in range(1,21):
    transitions_lopes=np.load('Mondes/Transitions_Lopes_no_stat '+str(number_world)+'.npy',allow_pickle=True)
    transitions_lopes_i_variable=np.load('Mondes/Transitions_Lopes_'+str(number_world)+'.npy',allow_pickle=True)
    environments_parameters["Lopes_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable}
    
    environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable,'transitions_no_stat':transitions_lopes}
    environments_parameters["Lopes_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable}
    all_environments["Lopes_nostat_{0}".format(number_world)]=Lopes_nostat
    all_environments["Lopes_{0}".format(number_world)]=Lopes_State

seed=173
np.random.seed(seed)
random.seed(seed)
agent_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.1},
            Rmax_Agent:{'gamma':0.95, 'm':7,'Rmax':1,'known_states':True,'u_m':7},
            BEB_Agent:{'gamma':0.95,'beta':3,'known_states':True,'coeff_prior':0.001,'informative':False},
            BEBLP_Agent:{'gamma':0.95,'beta':3,'step_update':5,'coeff_prior':0.001,'alpha':0.3},
            RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':0.6,'m':2.6},}


nb_iters=1
trials = 100
max_step =30
photos=[10,20,30]
screen=0
accuracy=0.01
pas_VI=50

#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agents={'BEBLP':BEBLP_Agent}
#environments=['Lopes_{0}'.format(num) for num in range(1,21)]+['Lopes_nostat_{0}'.format(num) for num in range(1,21)]

names_env = ['Lopes_nostat_1']
    
rewards={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
steps={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
exploration={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}
pol_error={(name_agent,name_environment):[] for name_agent in agents.keys() for name_environment in names_env}

for name_environment in names_env:   
    print(name_environment)
    for iteration in range(nb_iters):
        print(iteration)
        for name_agent,agent in agents.items(): 
            print(name_agent)
            environment=all_environments[name_environment](**environments_parameters[name_environment])
            
            globals()[name_agent]=agent(environment,**agent_parameters[agent]) #Defining a new agent from the dictionary agents
            
            reward,step_number,policy_value_error= play(environment,globals()[name_agent],trials=trials,max_step=max_step,photos=photos,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
            
            rewards[(name_agent,name_environment)].append(reward)
            steps[(name_agent,name_environment)].append(step_number)
            exploration[(name_agent,name_environment)].append(sum([len(value.keys()) for value in globals()[name_agent].counter.values()])/environment.max_exploration)
            pol_error[(name_agent,name_environment)].append(policy_value_error)
            

### Save results ###

temps=str(round(time.time()))

results={'seed':seed,'nb_iters':nb_iters,'trials':trials,'max_step':max_step,'agent_parameters':agent_parameters,'agents':agents,'environments':names_env,'rewards':rewards,'step_number':step_number,'pol_error':pol_error}

np.save('Results/'+temps+'_polerror.npy',pol_error)
save_pickle(results,'Results/'+temps+'.pickle')
test=open_pickle('Results/'+temps+'.pickle')


### Extracting results ###

#For each agent and each world

mean_rewards={(name_agent,name_environment): np.mean(rewards[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}

mean_exploration={(name_agent,name_environment): np.mean(exploration[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}
stats_convergence={(name_agent,name_environment):[convergence(rewards[(name_agent,name_environment)][nb_iter]) for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_stats={(name_agent,name_environment): np.average(stats_convergence[(name_agent,name_environment)],axis=0)for name_agent in agents.keys() for name_environment in names_env}


trial_plateau={(name_agent,name_environment):np.array(stats_convergence[(name_agent,name_environment)])[:,0] for name_agent in agents.keys() for name_environment in names_env}
step_plateau={(name_agent,name_environment):[steps[(name_agent,name_environment)][nb_iter][int(trial_plateau[(name_agent,name_environment)][nb_iter])] for nb_iter in range(nb_iters)]for name_agent in agents.keys() for name_environment in names_env}
avg_step_plateau={(name_agent,name_environment): np.average(step_plateau[(name_agent,name_environment)]) for name_agent in agents.keys() for name_environment in names_env}
min_length_agent={name_agent:np.min([len(pol_error[name_agent,name_environment][i]) for i in range(nb_iters) for name_environment in names_env]) for name_agent in agents.keys()}
min_length=np.min([len(pol_error[name_agent,name_environment][i]) for i in range(nb_iters) for name_agent in agents.keys() for name_environment in names_env])
mean_pol_error={(name_agent,name_environment):np.average([pol_error[name_agent,name_environment][i][:min_length_agent[name_agent]] for i in range(nb_iters)],axis=0) for name_environment in names_env for name_agent in agents.keys()}

#For each agent

mean_reward_agent={name_agent: np.mean([mean_rewards[(name_agent,name_environment)]for name_environment in names_env]) for name_agent in agents.keys() }

mean_exploration_agent={name_agent: np.mean([mean_exploration[(name_agent,name_environment)]for name_environment in names_env]) for name_agent in agents.keys() }

stats_agent={name_agent:np.average([avg_stats[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}
step_plateau_agent={name_agent: np.average([avg_step_plateau[(name_agent,name_environment)] for name_environment in names_env],axis=0) for name_agent in agents.keys()}

mean_pol_error_agent={name_agent: np.average([np.average([pol_error[name_agent,name_environment][i][:min_length_agent[name_agent]] for i in range(nb_iters)],axis=0) for name_environment in names_env],axis=0) for name_agent in agents.keys()}
#pol_error_agent={name_agent:np.average([pol_error[name_agent,name_environment][i][:min_length_agent[name_agent]] for name_environment in names_env for i in range(nb_iters)],axis=0) for name_agent in agents.keys()}
std_pol_error_agent={name_agent:np.std([pol_error[name_agent,name_environment][i][:min_length_agent[name_agent]] for name_environment in names_env for i in range(nb_iters)],axis=0)/np.sqrt(nb_iters*len(names_env)) for name_agent in agents.keys()}
print("")
for name_agent in agents.keys():
    print(name_agent+' : '+ 'avg_reward= '+str(round(mean_reward_agent[name_agent],2))+", trial_conv= "+str(stats_agent[name_agent][0])+
          ', step_conv= '+str(round(step_plateau_agent[name_agent]))+
          ', mean= '+str(round(stats_agent[name_agent][1],2))+', var= '+str(round(stats_agent[name_agent][2],2))+', explo= '+str(round(mean_exploration_agent[name_agent],2)))
    print("")


###Basic visualisation ###

rewards_agent_environment={(name_agent,name_environment): np.average(np.array(rewards[name_agent,name_environment]),axis=0) for name_environment in names_env for name_agent in agents.keys()}
rewards_agent={name_agent: np.average(np.average(np.array([rewards[name_agent,name_environment] for name_environment in names_env]),axis=1),axis=0) for name_agent in agents.keys()}
std_rewards_agent={name_agent: np.std(np.array([rewards[name_agent,name_environment] for name_environment in names_env]),axis=0)[0] for name_agent in agents.keys()}


    

rename={'RA':'R-max','BEB':'BEB','BEBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'Ɛ-greedy'}
colors={'RA':'royalblue','RALP':'royalblue','Epsilon_MB':'red','BEB':'black','BEBLP':'black'}
markers={'RA':'^','RALP':'o','BEB':'x','BEBLP':'*','Epsilon_MB':'s'}
linewidths={'RA':'0.75','RALP':'1.25','BEB':'0.75','BEBLP':'1.25','Epsilon_MB':'0.75'}
marker_sizes={'RA':'3','RALP':'3','BEB':'3','BEBLP':'3','Epsilon_MB':'3'}

fig=plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1)
for name_agent in agents.keys():
    plt.errorbar([pas_VI*i for i in range(min_length_agent[name_agent])],mean_pol_error_agent[name_agent], 
                 yerr=std_pol_error_agent[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
                 elinewidth=0.5,label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent],fillstyle='none',capsize=1)
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Results/pol_error'+temps+names_env[0]+'.png')
plt.show()

fig_reward=plt.figure(dpi=300)
ax_reward=fig_reward.add_subplot(1,1,1)
for name_agent in agents.keys():
    plt.errorbar([i+1 for i in range(trials)],rewards_agent[name_agent], 
                 yerr=std_rewards_agent[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
                 elinewidth=0.5,label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent],fillstyle='none',capsize=1)
plt.xlabel("Trial")
plt.ylabel("Reward")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Results/Rewards'+temps+names_env[0]+'.png')
plt.show()

fig_reward=plt.figure(dpi=300)
ax_reward=fig_reward.add_subplot(1,1,1)
for name_agent in agents.keys():
    plt.plot([i+1 for i in range(trials)],rewards_agent[name_agent], 
                 color=colors[name_agent],linewidth=linewidths[name_agent],
                 label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent],fillstyle='none')
plt.xlabel("Trial")
plt.ylabel("Reward")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Results/Rewards_nostd'+temps+names_env[0]+'.png')
plt.show()





