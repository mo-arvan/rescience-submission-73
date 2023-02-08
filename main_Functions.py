import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from multiprocessing import Pool

from Lopesworld import Lopes_environment

from policy_Functions import value_iteration,get_optimal_policy,policy_evaluation,get_agent_current_policy
from agents import Rmax,BEB,Epsilon_MB,EBLP,RmaxLP



def loading_environments():
    
    environments_parameters={}
    reward_0_1=np.load('Environments/Rewards_Lopes_1_-0.1.npy',allow_pickle=True)
    reward_1=np.load('Environments/Rewards_Lopes_1_-1.npy',allow_pickle=True)
    
    transitions_lopes=np.load('Environments/Transitions_Lopes_-0.1_1.npy',allow_pickle=True)
    environments_parameters['Lopes']={'transitions':transitions_lopes,'rewards':reward_0_1}
    
    for number_non_stationarity in range(1,21):
        transitions_non_stat_article=np.load('Environments/Transitions_non_stat_article-0.1_1_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
        transitions_strong_non_stat=np.load('Environments/Transitions_strong_non_stat_-0.1_1_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
        environments_parameters["Non_stat_article_-0.1_{0}".format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_0_1,'transitions_after_change':transitions_non_stat_article}
        environments_parameters["Non_stat_strong_-0.1_{0}".format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_0_1,'transitions_after_change':transitions_strong_non_stat}
    
    
    for number_world in range(1,11):
        transitions_lopes=np.load('Environments/Transitions_Lopes_-1_'+str(number_world)+'.npy',allow_pickle=True)
        environments_parameters['Stationary_Lopes_-1_'+str(number_world)]={'transitions':transitions_lopes,'rewards':reward_1}
        for number_non_stationarity in range(1,11):
            transitions_non_stat_article=np.load('Environments/Transitions_non_stat_article-1_'+str(number_world)+'_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
            transitions_strong_non_stat=np.load('Environments/Transitions_strong_non_stat_-1_'+str(number_world)+'_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
            environments_parameters["Non_stat_article_-1_{0}".format(number_world)+'_{0}'.format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_1,'transitions_after_change':transitions_non_stat_article}
            environments_parameters["Non_stat_strong_-1_{0}".format(number_world)+'_{0}'.format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_1,'transitions_after_change':transitions_strong_non_stat}
    
    return environments_parameters



def getting_simulations_to_do(*args):
    return [i for i in itertools.product(*args)]


### Play functions ###

def play(environment, name_agent, agent_parameters, trials=100, max_step=30, screen=False,photos=[10,20,50,80,99],accuracy_VI=1e-3,step_between_VI=50):
    
    agents={'R-max':Rmax,'ζ-R-max':RmaxLP,'BEB':BEB,'ζ-EB':EBLP,'ε-greedy':Epsilon_MB}
    agent=agents[name_agent](environment,**agent_parameters)
    
    
    reward_per_episode, optimal_policy_value_error, real_policy_value_error=[],[],[]
    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)

    for trial in range(trials):
        if screen : take_picture(agent,trial,environment,photos) 
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            if environment.total_steps==900:
                    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)
            if agent.step_counter%step_between_VI==0:
                optimal_policy_value_error.append(policy_evaluation(environment,get_optimal_policy(agent,agent.gamma,accuracy_VI),agent.gamma,accuracy_VI)[environment.first_location]-val_iteration[environment.first_location]) 
                real_policy_value_error.append(policy_evaluation(environment,get_agent_current_policy(agent),agent.gamma,accuracy_VI)[environment.first_location]-val_iteration[environment.first_location])
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , new_state = environment.make_step(action)            
            agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1
            if step==max_step:
                game_over = True
                environment.current_location=environment.first_location
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode,optimal_policy_value_error,real_policy_value_error



def one_parameter_play_function(all_params):
    return play_with_params(all_params[0],all_params[1],all_params[2],all_params[3])

def main_function(all_seeds,every_simulation,play_params,agent_parameters) :
    before=time.time()
    if type(agent_parameters)==dict : all_parameters= [[play_params,all_seeds[index_seed], agent_parameters,every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    else : 
        all_parameters=[[play_params,all_seeds[index_seed], agent_parameters[index_seed],every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    pool = Pool()
    results=pool.map(one_parameter_play_function,all_parameters)
    pool.close()
    pool.join()
    opt_pol_errors,rewards,real_pol_errors={},{},{}
    for result in results : 
        opt_pol_errors[result[0]]=result[1][1]
        rewards[result[0]]=result[1][0]
        real_pol_errors[result[0]]=result[1][2]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    return opt_pol_errors,real_pol_errors,rewards

def play_with_params(play_parameters,seed,agent_parameters,simulation_to_do):
    np.random.seed(seed)
    environment_parameters=loading_environments()
    name_environment,name_agent,iteration=simulation_to_do[:3]
    environment=Lopes_environment(**environment_parameters[name_environment])
    return simulation_to_do,play(environment,name_agent,agent_parameters[name_agent],**play_parameters)


### Extracting results ###

def get_mean_and_SEM(dictionary,names_environments,agents_tested,number_of_iterations):
    
    mean={name_agent: np.average([dictionary[name_environment,name_agent,i] for i in range(number_of_iterations) for name_environment in names_environments],axis=0) for name_agent in agents_tested}
    SEM={name_agent: np.std([dictionary[name_environment,name_agent,i] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) for name_agent in agents_tested}
    
    return (mean,SEM)

def extracting_results(rewards,opt_pol_error,real_pol_error,names_environments,agents_tested,number_of_iterations):
    
    optimal_statistics=get_mean_and_SEM(opt_pol_error,names_environments,agents_tested,number_of_iterations)
    real_statistics=get_mean_and_SEM(real_pol_error,names_environments,agents_tested,number_of_iterations)
    rewards_statistics=get_mean_and_SEM(rewards,names_environments,agents_tested,number_of_iterations)
    
    return optimal_statistics, real_statistics, rewards_statistics

def evaluate_agents(environments,agents,nb_iters,play_params,agent_parameters,starting_seed):
    every_simulation=getting_simulations_to_do(environments,agents,range(nb_iters))
    all_seeds=[starting_seed+i for i in range(len(every_simulation))]
    pol_errors_opti,pol_errors_real,rewards=main_function(all_seeds,every_simulation,play_params,agent_parameters)
    optimal, real, rewards =extracting_results(rewards,pol_errors_opti,pol_errors_real,environments,agents,nb_iters)
    save_and_plot(optimal, real, rewards ,agents,environments,play_params,environments,agent_parameters)


###Basic visualisation ###
    
def save_and_plot(optimal_stats,real_stats,rewards_stats,agents_tested,names_environments,play_parameters,environment_parameters,agent_parameters):
    
    pol_opti,SEM_pol_opti=optimal_stats
    pol_real,SEM_pol_real=real_stats
    reward,SEM_reward=rewards_stats
    time_end=str(round(time.time()%1e7))
    
    np.save('Results/'+time_end+'_polerror_opti.npy',pol_opti)
    np.save('Results/'+time_end+'_polerror_real.npy',pol_real)
    np.save('Results/'+time_end+'_rewards.npy',reward)
    
    colors={'R-max':'#9d02d7','ζ-R-max':'#0000ff','ε-greedy':"#ff7763",'BEB':"#ffac1e",'ζ-EB':"#009435"}
    marker_sizes={'R-max':'2','ζ-R-max':'2','BEB':'2','ζ-EB':'2','ε-greedy':'2'}  
    markers={'R-max':'^','ζ-R-max':'o','BEB':'x','ζ-EB':'*','ε-greedy':'s'}
    
    length_pol=(play_parameters["trials"]*play_parameters["max_step"])//play_parameters["step_between_VI"]
    
    plot1D([-12.5,0.5],"Steps","Policy value error")  
    plot_agents(agents_tested,pol_opti,SEM_pol_opti,[i*play_parameters["step_between_VI"] for i in range(length_pol)],colors,markers,marker_sizes)
    plt.savefig('Results/pol_error_opti'+time_end+names_environments[0]+'.pdf',bbox_inches = 'tight')

    plot1D([-12.5,0.5],"Steps","Policy value error")
    plot_agents(agents_tested,pol_real,SEM_pol_real,[i*play_parameters["step_between_VI"] for i in range(length_pol)],colors,markers,marker_sizes)
    plt.savefig('Results/pol_error_real'+time_end+names_environments[0]+'.pdf',bbox_inches = 'tight')
    plt.close()
    
    plot1D([-1,26],"Trials","Reward")
    plot_agents(agents_tested,reward,SEM_reward,[i for i in range(play_parameters["trials"])],colors,markers,marker_sizes)
    plt.savefig('Results/Rewards'+time_end+names_environments[0]+'.pdf',bbox_inches = 'tight')
    plt.close()
        
def plot1D(ylim,xlabel,ylabel):
    fig=plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle='--')
    plt.ylim(ylim)


def plot_agents(agents_tested,values,SEM_values,x_range,colors,markers,marker_sizes):
    for name_agent in agents_tested :
        yerr0 = values[name_agent] - SEM_values[name_agent]
        yerr1 = values[name_agent] + SEM_values[name_agent]

        plt.fill_between(x_range, yerr0, yerr1, color=colors[name_agent], alpha=0.2)
        
        plt.plot(x_range,values[name_agent], 
                     color=colors[name_agent],label=name_agent,ms=marker_sizes[name_agent],marker=markers[name_agent])
    plt.legend()

### PICTURES ###


import seaborn as sns 

def plot_V(table,policy_table,position):
    plt.subplot(position,aspect='equal')
    table=np.reshape(table,(5,5))
    policy_table=np.reshape(policy_table,(5,5))
    sns.heatmap(table, cmap='crest',cbar=False,annot=table,fmt='.1f',
                annot_kws={"size": 35 / (np.sqrt(len(table))+2.5)})
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if policy_table[i,j]==4:
                plt.gca().add_patch(plt.Circle((j+0.5, i+0.8), radius=0.07, color='black', fill=True))
            else :
                rotation={0:(0.5,0.9,0,-0.05),1:(0.5,0.65,0,0.05),2:(0.65,0.8,-0.05,0),3:(0.4,0.8,+0.05,0)}
                rotation_to_make=rotation[policy_table[i,j]]
                plt.arrow(j+rotation_to_make[0], i+rotation_to_make[1], rotation_to_make[2], rotation_to_make[3], head_width=0.12, head_length=0.12, fc='black', ec='black',linewidth=0.5,
          length_includes_head=False, shape='full', overhang=0,head_starts_at_zero=True)
    plt.xticks([])
    plt.yticks([])
    


def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    best_q_values,policies=get_max_Q_values_and_policy(agent.Q)
                    plt.figure(dpi=300)
                    if type(agent).__name__ == 'Epsilon_MB':
                        plot_V(best_q_values,policies,111)
                    if type(agent).__name__ in ['BEB','Rmax','EBLP','RmaxLP']:
                        if type(agent).__name__ in ['Rmax','RmaxLP']: bonus=agent.R_VI
                        else : bonus=agent.bonus
                        plot_V(best_q_values,policies,121)
                        best_values_bonus,best_policy_bonus=get_max_Q_values_and_policy(bonus)
                        plot_V(best_values_bonus,best_policy_bonus,122)
                    plt.savefig("Images/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".pdf",bbox_inches = 'tight')
                    plt.close()

        
def get_max_Q_values_and_policy(table):
    best_values=np.max(table,axis=1)
    best_actions=np.argmax(table+1e-5*np.random.random(table.shape),axis=1)    
    return best_values,best_actions
 

def compute_optimal_policies():
    environment_parameters=loading_environments()
    for name_environment in environment_parameters.keys():
        if 'transitions_after_change' in environment_parameters[name_environment].keys():
            environment_parameters[name_environment]['transitions']=environment_parameters[name_environment]['transitions_after_change']
        environment=Lopes_environment(**environment_parameters[name_environment])
        V,policy=value_iteration(environment,gamma=0.95,accuracy=1e-3)
        plt.figure(dpi=300)
        plot_V(V,policy,111)
        plt.savefig("Images/Optimal policy in each environment/VI_"+name_environment+".pdf",bbox_inches = 'tight')
        plt.close()

