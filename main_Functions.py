import numpy as np
import copy
import pygame
import pickle
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from Lopesworld import Lopes_environment


from Representation import World_Representation
from policy_Functions import value_iteration,get_optimal_policy,policy_evaluation,get_agent_current_policy


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

def getting_simulations_to_do(names_environments=[],agents_tested={},number_of_iterations=1):
    every_simulation=[]
    for name_environment in names_environments:   
        for name_agent,agent in agents_tested.items(): 
            for iteration in range(number_of_iterations):
                    every_simulation.append((name_environment,name_agent,agent,iteration))
    return every_simulation  

### Play functions ###

def play(environment, agent, trials=100, max_step=30, screen=False,photos=[10,20,50,80,99],accuracy_VI=0.01,step_between_VI=50):
    reward_per_episode, optimal_policy_value_error, real_policy_value_error=[],[],[]
    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)
    for trial in range(trials):
        if screen : take_picture(agent,trial,environment,photos) 
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            if environment.total_steps==900:
                    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)
            if agent.step_counter%step_between_VI==0:
                optimal_policy_value_error.append(policy_evaluation(environment,get_optimal_policy(agent,environment,agent.gamma,accuracy_VI),agent.gamma,accuracy_VI)[environment.first_location]-val_iteration[environment.first_location]) 
                real_policy_value_error.append(policy_evaluation(environment,get_agent_current_policy(agent,environment,agent.gamma,accuracy_VI),agent.gamma,accuracy_VI)[environment.first_location]-val_iteration[environment.first_location])
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

environment_parameters=loading_environments()

def play_with_params(name_environment,name_agent,agent,iteration,play_parameters,seed,agent_parameters):
    np.random.seed(seed)
    environment=Lopes_environment(**environment_parameters[name_environment])
    globals()[name_agent]=agent(environment,**agent_parameters[agent])
    return (name_agent,name_environment,iteration),play(environment,globals()[name_agent],**play_parameters)
    

def one_parameter_play_function(args):
    return play_with_params(args[0][0],args[0][1],args[0][2],args[0][3],args[1],args[2],args[3])

def main_function(all_seeds,every_simulation,play_params,agent_parameters) :
    before=time.time()
    all_parameters=[[every_simulation[seed],play_params,all_seeds[seed], agent_parameters] for seed in range(len(all_seeds))]
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

### Extracting results ###

def get_mean_and_CI(dictionary,names_environments,agents_tested,number_of_iterations):
    mean={name_agent: np.average([dictionary[name_agent,name_environment,i] for i in range(number_of_iterations) for name_environment in names_environments],axis=0) for name_agent in agents_tested}
    CI={name_agent:1.96*np.std([dictionary[name_agent,name_environment,i] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) for name_agent in agents_tested}
    return mean,CI

def extracting_results(rewards,opt_pol_error,real_pol_error,names_environments,agents_tested,number_of_iterations):
    
    mean_optimal_pol,CI_optimal_pol=get_mean_and_CI(opt_pol_error,names_environments,agents_tested,number_of_iterations)
    mean_real_pol,CI_real_pol=get_mean_and_CI(real_pol_error,names_environments,agents_tested,number_of_iterations)
    mean_rewards,CI_rewards=get_mean_and_CI(rewards,names_environments,agents_tested,number_of_iterations)
    return mean_optimal_pol,CI_optimal_pol,mean_real_pol,CI_real_pol,mean_rewards,CI_rewards

def evaluate_agents(environments,agents,nb_iters,play_params,agent_parameters,starting_seed):
    every_simulation=getting_simulations_to_do(environments,agents,nb_iters)
    all_seeds=[starting_seed+i for i in range(len(every_simulation))]
    pol_errors_opti,pol_errors_real,rewards=main_function(all_seeds,every_simulation,play_params,agent_parameters)
    pol_opti,CI_pol_opti, pol_real, CI_pol_real, reward, CI_reward=extracting_results(rewards,pol_errors_opti,pol_errors_real,environments,agents,nb_iters)
    save_and_plot(pol_opti,CI_pol_opti,pol_real,CI_pol_real,reward,CI_reward,agents,environments,play_params,environments,agent_parameters)


###Basic visualisation ###
    
def save_and_plot(pol_opti,CI_pol_opti,pol_real,CI_pol_real,reward,CI_reward,agents_tested,names_environments,play_parameters,environment_parameters,agent_parameters):
    time_end=str(round(time.time()%1e7))
    results={'play_parameters':play_parameters,'agent_parameters':agent_parameters,'agents':agents_tested,'environments':names_environments,'rewards':reward,'pol_error_opti':pol_opti,'pol_error_real':pol_real}
    np.save('Results/'+time_end+'_polerror_opti.npy',pol_opti)
    np.save('Results/'+time_end+'_polerror_real.npy',pol_real)
    save_pickle(results,'Results/'+time_end+'.pickle')
    
    rename={'RA':'R-max','BEB':'BEB','EBLP':r'$\zeta-EB','RALP':r'$\zeta$-R-max','Epsilon_MB':r'$\epsilon$-greedy'}
    colors={'RA':'#9d02d7','RALP':'#0000ff','Epsilon_MB':"#ff7763",'BEB':"#ffac1e",'EBLP':"#009435"}
    marker_sizes={'RA':'3','RALP':'3','BEB':'3','EBLP':'3','Epsilon_MB':'3'}
    
    markers={'RA':'^','RALP':'o','BEB':'x','EBLP':'*','Epsilon_MB':'s'}
    length_pol=(play_parameters["trials"]*play_parameters["max_step"])//play_parameters["step_between_VI"]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    basic_plot([-12.5,0.5],"Steps","Policy value error","Performance of the optimal policy on the learned world")  
    plot_agents(agents_tested,pol_opti,CI_pol_opti,[i*play_parameters["step_between_VI"] for i in range(length_pol)],colors,rename,markers,marker_sizes)
    plt.legend()
    plt.savefig('Results/pol_error_opti'+time_end+names_environments[0]+'.png')

    
    basic_plot([-12.5,0.5],"Steps","Policy value error","Performance of the agent policy")
    plot_agents(agents_tested,pol_real,CI_pol_real,[i*play_parameters["step_between_VI"] for i in range(length_pol)],colors,rename,markers,marker_sizes)
    plt.legend()
    plt.savefig('Results/pol_error_real'+time_end+names_environments[0]+'.png')
    
    
    basic_plot([-1,26],"Trials","Reward","Reward over time")
    plot_agents(agents_tested,reward,CI_reward,[i for i in range(play_parameters["trials"])],colors,rename,markers,marker_sizes)
    plt.legend()
    plt.savefig('Results/Rewards'+time_end+names_environments[0]+'.png')

def basic_plot(ylim,xlabel,ylabel,title):
    fig=plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle='--')
    plt.ylim(ylim)
    
def plot_agents(agents_tested,value_to_plot,CI_value,x_range,colors,rename,markers,marker_sizes):
    for name_agent in agents_tested.keys():
        yerr0 = value_to_plot[name_agent] - CI_value[name_agent]
        yerr1 = value_to_plot[name_agent] + CI_value[name_agent]

        plt.fill_between(x_range, yerr0, yerr1, color=colors[name_agent], alpha=0.2)
        
        plt.plot(x_range,value_to_plot[name_agent], 
                     color=colors[name_agent],label=rename[name_agent],ms=marker_sizes[name_agent],marker=markers[name_agent])

### PICTURES ###

def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    value=copy.deepcopy(agent.Q)
                    img=picture_world(environment,value)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','EBLP_Agent','RmaxLP_Agent']:
                        pygame.image.save(img.screen,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    else : pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    if type(agent).__name__ in ['Rmax_Agent','RmaxLP_Agent']: 
                        bonus=copy.deepcopy(agent.R)
                        img2=picture_world(environment,bonus,True,norm=agent.Rmax)
                    if type(agent).__name__ in ['BEB_Agent','EBLP_Agent']: 
                        bonus=copy.deepcopy(agent.bonus)
                        img2=picture_world(environment,bonus,True,norm=agent.beta)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','EBLP_Agent','RmaxLP_Agent']:
                        pygame.image.save(img2.screen,"Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png")
                        merging_two_images(environment,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png",
                                           "Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png",
                                           "Images/"+type(agent).__name__+"_"+type(environment).__name__+" Q_table (left) and bonus (right) "+str(trial)+".png")

def merging_two_images(environment,img1,img2,path):
    pygame.init()
    image1 = pygame.image.load(img1)
    image2 = pygame.image.load(img2)

    screen = pygame.Surface((environment.height*100+200,environment.height*50+100))
    
    screen.fill((0,0,0))   
    screen.blit(image1, (50,  50))
    screen.blit(image2, (environment.height*50+150, 50))

    pygame.image.save(screen,path)

        
def get_normalized_best_values(table,environment):
    max_every_state=np.zeros((environment.height,environment.width))
    action_every_state=dict()
    for state in table.keys():
        q_values = table[state]
        max_value_state=max(q_values.values())
        max_every_state[state]=max_value_state
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]=best_action
    mini=np.min(max_every_state)
    max_every_state-=mini
    maxi=np.max(max_every_state)
    if maxi !=0 : max_every_state/=maxi     
    return max_every_state,action_every_state
 
def picture_world(environment,table,bonus=False,norm=1):       
    max_Q,best_actions=get_normalized_best_values(table,environment)
    if bonus : max_Q/=norm
    return World_Representation(environment,max_Q,best_actions)


def plot_VI(environment,gamma,accuracy): #only in gridworlds
    V,action=value_iteration(environment,gamma,accuracy)
    V_2=np.zeros((environment.height,environment.width))
    for state,value in V.items():
        V_2[state]=value
        if V_2[state]<0:V_2[state]=0
    V_2=V_2-np.min(V_2)
    max_value=np.max(V_2)
    V_2=V_2/max_value
    return World_Representation(environment,V_2,action)

def compute_optimal_policies():
    environment_parameters=loading_environments()
    for name_environment in environment_parameters.keys():
        if 'transitions_after_change' in environment_parameters[name_environment].keys():
            environment_parameters[name_environment]['transitions']=environment_parameters[name_environment]['transitions_after_change']
        environment=Lopes_environment(**environment_parameters[name_environment])
        gridworld=plot_VI(environment,gamma=0.95,accuracy=0.001)
        pygame.image.save(gridworld.screen,"Images/Optimal policy in each environment/VI_"+name_environment+".png")



### SAVING PARAMETERS ####

def save_pickle(dictionnaire,path):
    with open(path, 'wb') as f:
        pickle.dump(dictionnaire, f) 

def open_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)



    


