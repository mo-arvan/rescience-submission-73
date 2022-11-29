import numpy as np
import copy
import pygame
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool


from Lopesworld import Lopes_State


from Representation import Graphique
from policy_Functions import value_iteration,get_optimal_policy,policy_evaluation


def loading_environments():
    environments_parameters={}
    reward_0_1=np.load('Mondes/Rewards_Lopes_1_-0.1.npy',allow_pickle=True)
    reward_1=np.load('Mondes/Rewards_Lopes_1_-1.npy',allow_pickle=True)
    transitions_lopes=np.load('Mondes/Transitions_Lopes_-0.1_1.npy',allow_pickle=True)
    environments_parameters['Lopes']={'transitions':transitions_lopes,'rewards':reward_0_1}
    for number_non_stationarity in range(1,21):
        transitions_non_stat_article=np.load('Mondes/Transitions_non_stat_article-0.1_1_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
        transitions_strong_non_stat=np.load('Mondes/Transitions_strong_non_stat_-0.1_1_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
        environments_parameters["Non_stat_article_-0.1_{0}".format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_0_1,'transitions_after_change':transitions_non_stat_article}
        environments_parameters["Non_stat_strong_-0.1_{0}".format(number_non_stationarity)]={'transitions':transitions_lopes,'rewards':reward_0_1,'transitions_after_change':transitions_strong_non_stat}
    for number_world in range(1,11):
        transitions_lopes=np.load('Mondes/Transitions_Lopes_-1_'+str(number_world)+'.npy',allow_pickle=True)
        for number_non_stationarity in range(1,11):
            transitions_non_stat_article=np.load('Mondes/Transitions_non_stat_article-1_'+str(number_world)+'_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
            transitions_strong_non_stat=np.load('Mondes/Transitions_strong_non_stat_-1_'+str(number_world)+'_'+str(number_non_stationarity)+'.npy',allow_pickle=True)
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
    reward_per_episode, policy_value_error=[],[]
    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)
    for trial in range(trials):
        if screen : take_picture(agent,trial,environment,photos) 
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            if environment.total_steps==900:
                    val_iteration,_=value_iteration(environment,agent.gamma,accuracy_VI)
            if agent.step_counter%step_between_VI==0:
                policy_value_error.append(policy_evaluation(environment,get_optimal_policy(agent,environment,agent.gamma,accuracy_VI),agent.gamma,accuracy_VI)[environment.first_location]-val_iteration[environment.first_location]) 
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
    return reward_per_episode,policy_value_error

environment_parameters=loading_environments()

def play_with_params(name_environment,name_agent,agent,iteration,play_parameters,seed,agent_parameters):
    np.random.seed(seed)
    environment=Lopes_State(**environment_parameters[name_environment])
    globals()[name_agent]=agent(environment,**agent_parameters[agent])
    return (name_agent,name_environment,iteration),play(environment,globals()[name_agent],**play_parameters)
    

def one_parameter_play_function(args):
    return play_with_params(args[1][0],args[1][1],args[1][2],args[1][3],args[0],args[2],args[3])

def main_function(all_seeds,every_simulation,play_params,agent_parameters) :
    before=time.time()
    all_parameters=[[play_params,every_simulation[seed],all_seeds[seed], agent_parameters] for seed in range(len(all_seeds))]
    pool = Pool()
    results=pool.map(one_parameter_play_function,all_parameters)
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
    
def save_and_plot(pol,CI_pol,reward,CI_reward,agents_tested,names_environments,play_parameters,environment_parameters,agent_parameters):
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

        plt.fill_between([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent]))], yerr0, yerr1, color=colors[name_agent], alpha=0.2)

        plt.plot([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent]))],pol[name_agent],color=colors[name_agent],linewidth=linewidths[name_agent],
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





### PICTURES ###

def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    value=copy.deepcopy(agent.Q)
                    img=picture_world(environment,value)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','BEBLP_Agent','RmaxLP_Agent']:
                        pygame.image.save(img.screen,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    else : pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    if type(agent).__name__ in ['Rmax_Agent','RmaxLP_Agent']: bonus=copy.deepcopy(agent.R)
                    if type(agent).__name__ in ['BEB_Agent','BEBLP_Agent']: bonus=copy.deepcopy(agent.bonus)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','BEBLP_Agent','RmaxLP_Agent']:
                        img2=picture_world(environment,bonus)
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

        
def normalized_table(table,environment):
    max_every_state=np.zeros((environment.height,environment.width))
    action_every_state=dict()
    for state in table.keys():
        q_values = table[state]
        max_value_state=max(q_values.values())
        max_every_state[state]=max_value_state
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]=best_action
    mini,maxi=np.min(max_every_state),np.max(max_every_state)
    if mini < 0 : max_every_state-=mini
    if maxi !=0 : max_every_state/=maxi     
    return max_every_state,action_every_state
 
def picture_world(environment,table):       
    max_Q,best_actions=normalized_table(table,environment)
    return Graphique(environment,max_Q,best_actions)


def plot_VI(environment,gamma,accuracy): #only in gridworlds
    V,action=value_iteration(environment,gamma,accuracy)
    V_2=np.zeros((environment.height,environment.width))
    for state,value in V.items():
        V_2[state]=value
        if V_2[state]<0:V_2[state]=0
    V_2=V_2-np.min(V_2)
    max_value=np.max(V_2)
    V_2=V_2/max_value
    return Graphique(environment,V_2,action)

def compute_optimal_policies():
    environment_parameters=loading_environments()
    for name_environment in environment_parameters.keys():
        if 'transitions_after_change' in environment_parameters[name_environment].keys():
            environment_parameters[name_environment]['transitions']=environment_parameters[name_environment]['transitions_after_change']
        environment=Lopes_State(**environment_parameters[name_environment])
        gridworld=plot_VI(environment,gamma=0.95,accuracy=0.001)
        pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_"+name_environment+".png")


### SAVING PARAMETERS ####

def save_pickle(dictionnaire,path):
    with open(path, 'wb') as f:
        pickle.dump(dictionnaire, f) 

def open_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

##### EXTRACTING RESULTS ####



######## VISUALISATION ########

def convert_from_default(dic):
    from collections import defaultdict
    if isinstance(dic,defaultdict):
        return dict((key,convert_from_default(value)) for key,value in dic.items())
    else : return dic


def plot3D(table_3D,x_name='beta',y_name='gamma'):
    dataframe=pd.DataFrame({x_name:table_3D[:,0], y_name:table_3D[:,1], 'values':table_3D[:,2]})
    data_pivoted = dataframe.pivot(x_name, y_name, "values")
    sns.heatmap(data_pivoted,cmap='Blues')


####### GET STATS #####

def convergence(array,longueur=30,variation=0.2,absolu=0.1,artefact=3):
    for i in range(len(array)-longueur):
        table=array[i:i+longueur]
        mean=np.mean(table)
        if mean >0:
            mauvais=0
            valid=True
            for element in table : 
                if element < (1-variation)*mean-absolu>element or element > (1+variation)*mean+absolu:
                    mauvais+=1
                    if mauvais > artefact:
                        valid=False
            if valid : 
                return [len(array)-len(array[i+1:]),np.mean(array[i+1:]),np.var(array[i+1:])]            
    return [len(array)-1,np.mean(array),np.var(array)]

    


