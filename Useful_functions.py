import numpy as np
import copy
import pygame
import seaborn as sns
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

from Gridworld import State
from Complexworld import ComplexState
from Deterministic_world import Deterministic_State
from Uncertain_world import Uncertain_State
from Lopesworld import Lopes_State
from Two_step_task import Two_step
from Lopes_nonstat import Lopes_nostat
from Deterministic_nostat import Deterministic_no_stat
from Uncertainworld_U import Uncertain_State_U
from Uncertainworld_B import Uncertain_State_B

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_sum import Kalman_agent_sum
from Rmax import Rmax_Agent
from BEB import BEB_Agent
from BEBLP2 import BEBLP_Agent
from RmaxLP import RmaxLP_Agent
from greedyMB import QMB_Agent
from PEI_E import PEI_E_Agent
from PIM import PIM_Agent

from Representation import Graphique



#MAIN

def play(environment, agent, trials=200, max_step=500, screen=0,photos=[10,20,50,100,199,300,499],accuracy=0.01,pas_VI=50):
    reward_per_episode = []
    step_number=[]
    policy_value_error=[]
    pol_updated=False
    val_iteration,_=value_iteration(environment,agent.gamma,accuracy)
    for trial in range(trials):
        if screen : take_picture(agent,trial,environment,photos) #Visualisation
        
        cumulative_reward, step, game_over= 0,0,False
        while not game_over :
            if type(environment).__name__ in ['Uncertain_State_U','Uncertain_State_B','Deterministic_no_stat','Lopes_nostat'] and not pol_updated:
                if environment.changed :
                    pol_updated=True
                    val_iteration,_=value_iteration(environment,agent.gamma,accuracy)
            if agent.step_counter%pas_VI==0:
                policy_value_error.append(policy_evaluation(environment,get_policy_2(agent,environment,agent.gamma,accuracy),agent.gamma,accuracy)[environment.first_location]-val_iteration[environment.first_location]) 
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , terminal = environment.make_step(action) #reward and if state is terminal
            new_state = environment.current_location            
            agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1
            if terminal == True or step==max_step:
                game_over = True
                environment.current_location=environment.first_location
                step_number.append(agent.step_counter)
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode,step_number,policy_value_error

### Initializing environments ###

environments_parameters={'Two_Step':{},'Lopes':{'transitions':np.load('Mondes/Transitions_Lopes.npy',allow_pickle=True)}}
all_environments={'Lopes':Lopes_State,'Two_Step':Two_step}
for number_world in range(1,21):
    world=np.load('Mondes/World_'+str(number_world)+'.npy')
    world_2=np.load('Mondes/World_'+str(number_world)+'_B.npy')
    transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
    transitions_U=np.load('Mondes/Transitions_'+str(number_world)+'_U.npy',allow_pickle=True)
    transitions_B=np.load('Mondes/Transitions_'+str(number_world)+'_B.npy',allow_pickle=True)
    transitions_lopes=np.load('Mondes/Transitions_Lopes_non_stat'+str(number_world)+'.npy',allow_pickle=True)
    transitions_lopes_i_variable=np.load('Mondes/Transitions_Lopes_'+str(number_world)+'.npy',allow_pickle=True)
    transitions_lopes_optimal=np.load('Mondes/Transitions_Lopes_non_stat_optimal'+str(number_world)+'.npy',allow_pickle=True)
    environments_parameters["D_{0}".format(number_world)] = {'world':world}
    environments_parameters["U_{0}".format(number_world)] = {'world':world,'transitions':transitions}
    environments_parameters["DB_{0}".format(number_world)] = {'world':world,'world2':world_2}
    environments_parameters["UU_{0}".format(number_world)] = {'world':world,'transitions':transitions,'transitions_U':transitions_U}
    environments_parameters["UB_{0}".format(number_world)] = {'world':world,'world2':world_2,'transitions':transitions,'transitions_B':transitions_B} 
    environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable,'transitions2':transitions_lopes}
    environments_parameters["Lopes_nostat_optimal_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable,'transitions2':transitions_lopes_optimal}
    environments_parameters["Lopes_{0}".format(number_world)]={'transitions':transitions_lopes_i_variable}
    all_environments["D_{0}".format(number_world)]=Deterministic_State
    all_environments["U_{0}".format(number_world)]=Uncertain_State
    all_environments["DB_{0}".format(number_world)]=Deterministic_no_stat
    all_environments["UB_{0}".format(number_world)]=Uncertain_State_B
    all_environments["UU_{0}".format(number_world)]=Uncertain_State_U
    all_environments["Lopes_nostat_{0}".format(number_world)]=Lopes_nostat
    all_environments["Lopes_nostat_optimal_{0}".format(number_world)]=Lopes_nostat
    all_environments["Lopes_{0}".format(number_world)]=Lopes_State

### PICTURES ###

def take_picture(agent,trial,environment,photos):
            if trial in photos:
                    value=copy.deepcopy(agent.Q)
                    img=picture_world(environment,value)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','Kalman_agent_sum','BEBLP_Agent','RmaxLP_Agent']:
                        pygame.image.save(img.screen,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    else : pygame.image.save(img.screen,"Images/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png")
                    if type(agent).__name__ =='Kalman_agent_sum': bonus=copy.deepcopy(agent.K_var)
                    if type(agent).__name__=='Rmax_Agent': bonus=copy.deepcopy(agent.R)
                    if type(agent).__name__=='BEB_Agent': bonus=copy.deepcopy(agent.bonus)
                    if type(agent).__name__=='BEBLP_Agent': bonus=copy.deepcopy(agent.bonus)
                    if type(agent).__name__=='RmaxLP_Agent': bonus=copy.deepcopy(agent.R)
                    if type(agent).__name__ in ['BEB_Agent','Rmax_Agent','Kalman_agent_sum','BEBLP_Agent','RmaxLP_Agent']:
                        img2=picture_world(environment,bonus)
                        pygame.image.save(img2.screen,"Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png")
                        merging_two_images(environment,"Images/Solo/"+type(agent).__name__+"_"+type(environment).__name__+"_"+str(trial)+".png","Images/Solo/"+type(agent).__name__+"_bonus"+"_"+type(environment).__name__+str(trial)+".png","Images/"+type(agent).__name__+"_"+type(environment).__name__+" Q_table (left) and bonus (right) "+str(trial)+".png")

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
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 45
    cell_height = 45
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.reward_states,init_loc,max_Q,best_actions)
    return gridworld

### SAVING PARAMETERS ####

def save_pickle(dictionnaire,path):
    with open(path, 'wb') as f:
        pickle.dump(dictionnaire, f) 

def open_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

##### EXTRACTING RESULTS ####


def extracting_results(results):
    pass
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
    

def plot4D(table_4D,x_name='alpha',y_name='beta',z_name='gamma',save=False): #Changer en %matplotlib auto
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(table_4D[:,0], table_4D[:,1], table_4D[:,2], c=table_4D[:,3],s=50,cmap='Blues')
    fig.colorbar(img,location='left')
    ax.set_xlabel(x_name), ax.set_ylabel(y_name), ax.set_zlabel(z_name)
    fig.show()    
    
    values=table_4D[:,3]
    color=[]
    for value in values : 
        c=0
        if value > 100 : c=100
        elif value>50: c = 50
        elif value >15 : c=15
        color.append(c)
    table2=table_4D[table_4D[:,3]>15]
    color=np.array(color)
    color=color[color>0]
    
    fig2 = plt.figure()    
    ax = fig2.add_subplot(111, projection='3d')
    img = ax.scatter(table2[:,0], table2[:,1], table2[:,2], c=color,s=50,cmap=plt.get_cmap('Set3', 3))
    clb=fig2.colorbar(img,location='left',ticks=np.array([30,57.8,85]))
    clb.set_ticklabels(['> 15','> 50','> 100'])
    ax.set_xlabel(x_name), ax.set_ylabel(y_name), ax.set_zlabel(z_name)
    #save_interactive(fig2,str(time.time()))
    fig2.show()
    print(table_4D[table_4D[:,3]>50])

    
def save_interactive(fig,name):
    pickle.dump(fig, open('Interactive/'+name+'.fig.pickle', 'wb'))

def plot_interactive(name_fig):
    figx=pickle.load(open('Interactive/'+name_fig,'rb'))
    figx.show()

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

    
            
###### Policy evaluation #####

def value_iteration(environment,gamma,accuracy):
    V={state:1 for state in environment.states}
    policy={state:0 for state in environment.states}
    delta=accuracy+1
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            V[state]=np.max([np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
            delta=max(delta,np.abs(value_V-V[state]))
    for state,value in V.items():
        policy[state]=np.argmax([np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
    return V,policy


def plot_VI(environment,gamma,accuracy): #only in gridworlds
    V,action=value_iteration(environment,gamma,accuracy)
    V_2=np.zeros((environment.height,environment.width))
    for state,value in V.items():
        V_2[state]=value
        if V_2[state]<0:V_2[state]=0
    max_value=np.max(V_2)
    V_2=V_2/max_value
    init_loc=[environment.first_location[0],environment.first_location[1]]
    screen_size = environment.height*50
    cell_width = 45
    cell_height = 45
    cell_margin = 5
    gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,environment.grid,environment.reward_states,init_loc,V_2,action)
    pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_test"+type(environment).__name__+".png")
    return gridworld

def policy_evaluation(environment,policy,gamma,accuracy):
    V={state:1 for state in environment.states}
    delta=accuracy+1
    for state in V.keys():
        if state not in policy.keys():
            policy[state]=np.random.choice(environment.actions)
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            actions=policy[state]
            V[state]=np.sum([probability_action*np.sum([environment.transitions[action][state][new_state]*(environment.values[state[0],state[1],action]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action,probability_action in actions.items()])
            delta=max(delta,np.abs(value_V-V[state]))
    return V


def policy_evaluation2(environment,policy,gamma,accuracy):
    V={state:1 for state in environment.states}
    delta=accuracy+1
    for state in V.keys():
        if state not in policy.keys():
            policy[state]=np.random.choice(environment.actions)
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            actions=policy[state]
            V[state]=np.sum([probability_action*np.sum([environment.transitions[action][state][new_state]*(environment.values[new_state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action,probability_action in actions.items()])
            delta=max(delta,np.abs(value_V-V[state]))
    return V

def get_policy(agent):
    action_every_state=dict()
    for state in agent.Q.keys():
        q_values = agent.Q[state]
        max_value_state=max(q_values.values())
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]={best_action:1}
        if type(agent).__name__=='QMB_Agent':
            random_actions=list(range(5))
            del random_actions[best_action]
            action_every_state[state]={other_action: agent.epsilon/5 for other_action in random_actions}
            action_every_state[state][best_action]=1-agent.epsilon+agent.epsilon/5
    return action_every_state

def get_policy_2(agent,environment,gamma,accuracy):
    transitions=agent.tSAS
    rewards=agent.R
    V={state:1 for state in transitions}
    policy=dict()
    delta=accuracy+1
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            V[state]=np.max([np.sum([transitions[state][action][new_state]*(rewards[state][action]+gamma*V[new_state]) for new_state in transitions[state][action].keys()]) for action in environment.actions])
            delta=max(delta,np.abs(value_V-V[state]))
    for state,value in V.items():
        policy[state]={np.argmax([np.sum([transitions[state][action][new_state]*(rewards[state][action]+gamma*V[new_state]) for new_state in transitions[state][action]]) for action in environment.actions]):1}
    return policy

    
    

##Optimal policies ###

def compute_optimal_policies(environments_parameters=environments_parameters):
    for name_environment in all_environments.keys():
        environment=all_environments[name_environment](**environments_parameters[name_environment])
        for number_world in range(1,21):
            world=np.load('Mondes/World_'+str(number_world)+'.npy')
            world_2=np.load('Mondes/World_'+str(number_world)+'_B.npy')
            transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)
            transitions_U=np.load('Mondes/Transitions_'+str(number_world)+'_U.npy',allow_pickle=True)
            transitions_B=np.load('Mondes/Transitions_'+str(number_world)+'_B.npy',allow_pickle=True)
            transitions_lopes=np.load('Mondes/Transitions_Lopes_non_stat'+str(number_world)+'.npy',allow_pickle=True)
            environments_parameters["DB_{0}".format(number_world)] = {'world':world_2,'world2':world}
            environments_parameters["UU_{0}".format(number_world)] = {'world':world,'transitions':transitions_U,'transitions_U':transitions}
            environments_parameters["UB_{0}".format(number_world)] = {'world':world_2,'world2':world,'transitions':transitions_B,'transitions_B':transitions_lopes_i_variable} 
            environments_parameters["Lopes_nostat_optimal_{0}".format(number_world)]={'transitions':transitions_lopes_optimal,'transitions2':transitions_lopes_i_variable}
            environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':transitions_lopes,'transitions2':transitions}
        if type(environment).__name__ !='Two_step':
            gridworld=plot_VI(environment,gamma=0.95,accuracy=0.001)
            pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_"+name_environment+".png")

### Parametter fitting ##

precision_conv=-0.2

def fitting_BEB(environment_names,betas,priors,trials = 300,max_step = 30,accuracy=0.05,screen=0,pas_VI=25,informative=True):
    BEB_parameters={(beta,prior):{'gamma':0.95,'beta':beta,'known_states':True,'coeff_prior':prior,'informative':informative} for beta in betas for prior in priors}
    pol_error={(beta,prior):[] for beta in betas for prior in priors}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for beta in betas :
            print(beta)
            for prior in priors :
                BEB=BEB_Agent(environment,**BEB_parameters[(beta,prior)]) #Defining a new agent from the dictionary agents
                
                _,step_number,policy_value_error= play(environment,BEB,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[beta,prior].append(policy_value_error)

    min_length_param={(beta,prior):np.min([len(pol_error[beta,prior][i]) for i in range(len(environment_names))]) for beta in betas for prior in priors}
    mean_pol_error={(beta,prior): np.average([pol_error[beta,prior][i][:min_length_param[beta,prior]] for i in range(len(environment_names))],axis=0) for beta in betas for prior in priors}
    convergence_trial={(beta,prior):np.where(mean_pol_error[beta,prior]<precision_conv)[-1][-1] for beta in betas for prior in priors}
    convergence_step={(beta,prior):pas_VI*(convergence_trial[beta,prior]+1) for beta in betas for prior in priors}
    
    return mean_pol_error,convergence_step




def fitting_RALP(environment_names,alphas,ms,trials = 200,max_step = 30,accuracy=.05,screen=0,pas_VI=25):
    RALP_parameters={(alpha,m):{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':alpha,'m':m} for alpha in alphas for m in ms}
    pol_error={(alpha,m): [] for alpha in alphas for m in ms}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for alpha in alphas :
            print(alpha)
            for m in ms :
                RALP=RmaxLP_Agent(environment,**RALP_parameters[(alpha,m)]) #Defining a new agent from the dictionary agents
                
                reward,step_number,policy_value_error= play(environment,RALP,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[alpha,m].append(policy_value_error)

    min_length_param={(alpha,m):np.min([len(pol_error[alpha,m][i]) for i in range(len(environment_names))]) for alpha in alphas for m in ms}
    mean_pol_error={(alpha,m): np.average([pol_error[alpha,m][i][:min_length_param[alpha,m]] for i in range(len(environment_names))],axis=0) for alpha in alphas for m in ms}
    convergence_trial={(alpha,m):np.where(mean_pol_error[alpha,m]<precision_conv)[-1][-1] for alpha in alphas for m in ms}
    convergence_step={(alpha,m):pas_VI*(convergence_trial[alpha,m]+1) for alpha in alphas for m in ms}  
    
    return mean_pol_error,convergence_step



def fitting_QMB(environment_names,epsilons,trials=50,max_step=30,accuracy=0.01,screen=0,pas_VI=25):
    QMB_parameters={epsilon:{'gamma':0.95,'known_states':True,'epsilon':epsilon} for epsilon in epsilons}
    pol_error={epsilon:[] for epsilon in epsilons}
    
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for epsilon in epsilons:
            print(epsilon)
            QMB=QMB_Agent(environment,**QMB_parameters[epsilon]) #Defining a new agent from the dictionary agents
                
            _,step_number,policy_value_error= play(environment,QMB,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                    
            pol_error[epsilon].append(policy_value_error)

    min_length_param={epsilon:np.min([len(pol_error[epsilon][i]) for i in range(len(environment_names))]) for epsilon in epsilons}
    mean_pol_error={epsilon: np.average([pol_error[epsilon][i][:min_length_param[epsilon]] for i in range(len(environment_names))],axis=0) for epsilon in epsilons}
    convergence_trial={epsilon:np.where(mean_pol_error[epsilon]<precision_conv)[-1][-1] for epsilon in epsilons}
    convergence_step={epsilon:pas_VI*(convergence_trial[epsilon]+1) for epsilon in epsilons}  
    return mean_pol_error,convergence_step


def fitting_BEBLP(environment_names,betas,alphas,trials = 300,max_step = 30,accuracy=5,screen=0,pas_VI=25):
    BEBLP_parameters={(beta,alpha):{'gamma':0.95,'beta':beta,'alpha':alpha,'coeff_prior':0.001,'step_update':10} for beta in betas for alpha in alphas}
    pol_error={(beta,alpha):[] for beta in betas for alpha in alphas}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for beta in betas :
            print(beta)
            for alpha in alphas :
                BEBLP=BEBLP_Agent(environment,**BEBLP_parameters[(beta,alpha)]) #Defining a new agent from the dictionary agents
                
                _,step_number,policy_value_error= play(environment,BEBLP,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[beta,alpha].append(policy_value_error)

    min_length_param={(beta,alpha):np.min([len(pol_error[beta,alpha][i]) for i in range(len(environment_names))]) for beta in betas for alpha in alphas}
    mean_pol_error={(beta,alpha): np.average([pol_error[beta,alpha][i][:min_length_param[beta,alpha]] for i in range(len(environment_names))],axis=0) for beta in betas for alpha in alphas}
    convergence_trial={(beta,alpha):np.where(mean_pol_error[beta,alpha]<precision_conv)[-1][-1] for beta in betas for alpha in alphas}
    convergence_step={(beta,alpha):pas_VI*(convergence_trial[beta,alpha]+1) for beta in betas for alpha in alphas}
    
    return mean_pol_error,convergence_step


def fitting_RA(environment_names,u_ms,ms,trials = 200,max_step = 30,accuracy=.05,screen=0,pas_VI=25):
    RALP_parameters={(m,u_m):{'gamma':0.95,'Rmax':1,'m':m,'u_m':u_m} for u_m in u_ms for m in ms}
    pol_error={(m,u_m): [] for m in ms for u_m in u_ms}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for u_m in u_ms :
            print(u_m)
            for m in ms :
                RA=Rmax_Agent(environment,**RALP_parameters[(m,u_m)]) #Defining a new agent from the dictionary agents
                
                r_,step_number,policy_value_error= play(environment,RA,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment

                pol_error[m,u_m].append(policy_value_error)
    min_length_param={(m,u_m):np.min([len(pol_error[m,u_m][i]) for i in range(len(environment_names))]) for u_m in u_ms for m in ms}
    mean_pol_error={(m,u_m): np.average([pol_error[m,u_m][i][:min_length_param[m,u_m]] for i in range(len(environment_names))],axis=0) for u_m in u_ms for m in ms}
    convergence_trial={(m,u_m):np.where(mean_pol_error[m,u_m]<precision_conv)[-1][-1] for m in ms for u_m in u_ms}
    convergence_step={(m,u_m):pas_VI*(convergence_trial[m,u_m]+1) for m in ms for u_m in u_ms} 
    return mean_pol_error,convergence_step

def fitting_EGE(environment_names,gamma_es,coeff_es,trials = 100,max_step = 30,accuracy=.05,screen=0,pas_VI=50):
    EGE_parameters={(gamma_e,coeff_e):{'gamma':0.95,'gamma_e':gamma_e,'coeff_e':coeff_e,'epsilon':0.1} for gamma_e in gamma_es for coeff_e in coeff_es}
    pol_error={(gamma_e,coeff_e): [] for gamma_e in gamma_es for coeff_e in coeff_es}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for gamma_e in gamma_es :
            print(gamma_e)
            for coeff_e in coeff_es :
                EGE=EGE_Agent(environment,**EGE_parameters[(gamma_e,coeff_e)]) #Defining a new agent from the dictionary agents
                
                _,_,policy_value_error= play(environment,EGE,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment

                pol_error[gamma_e,coeff_e].append(policy_value_error)
    min_length_param={(gamma_e,coeff_e):np.min([len(pol_error[gamma_e,coeff_e][i]) for i in range(len(environment_names))]) for gamma_e in gamma_es for coeff_e in coeff_es}
    mean_pol_error={(gamma_e,coeff_e): np.average([pol_error[gamma_e,coeff_e][i][:min_length_param[gamma_e,coeff_e]] for i in range(len(environment_names))],axis=0) for gamma_e in gamma_es for coeff_e in coeff_es}
    convergence_trial={(gamma_e,coeff_e):np.where(mean_pol_error[gamma_e,coeff_e]<precision_conv)[-1][-1] for gamma_e in gamma_es for coeff_e in coeff_es}
    convergence_step={(gamma_e,coeff_e):pas_VI*(convergence_trial[gamma_e,coeff_e]+1) for gamma_e in gamma_es for coeff_e in coeff_es} 
    return mean_pol_error,convergence_step

def fitting_PIM(environment_names,alphas,betas, trials = 100,max_step = 30,accuracy=.01,screen=0,pas_VI=50):
    PIM_parameters={(alpha,beta):{'gamma':0.95,'alpha':alpha,'beta':beta,'k':2} for alpha in alphas for beta in betas}
    pol_error={(alpha,beta): [] for alpha in alphas for beta in betas}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for alpha in alphas :
            print(alpha)
            for beta in betas :
                PIM=PIM_Agent(environment,**PIM_parameters[(alpha,beta)]) #Defining a new agent from the dictionary agents
                
                _,_,policy_value_error= play(environment,PIM,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment

                pol_error[alpha,beta].append(policy_value_error)
    min_length_param={(alpha,beta):np.min([len(pol_error[alpha,beta][i]) for i in range(len(environment_names))]) for alpha in alphas for beta in betas}
    mean_pol_error={(alpha,beta): np.average([pol_error[alpha,beta][i][:min_length_param[alpha,beta]] for i in range(len(environment_names))],axis=0) for alpha in alphas for beta in betas}
    convergence_trial={(alpha,beta):np.where(mean_pol_error[alpha,beta]<precision_conv)[-1][-1] for alpha in alphas for beta in betas}
    convergence_step={(alpha,beta):pas_VI*(convergence_trial[alpha,beta]+1) for alpha in alphas for beta in betas} 
    return mean_pol_error,convergence_step




pas_VI=50
accuracy=0.01
trials=125
max_step=40


environment_names=['U_{0}'.format(num) for num in range(1,6)]

temps=str(time.time())
colors_6=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
markers=['^','o','x','*','s','P','.','D','1','v',',']

#E-GREEDY 

"""
epsilons=[0.005,0.1,0.2,0.5,0.8,1]
a,conv_a=fitting_QMB(environment_names,epsilons,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/e-greedy'+temps,a)
plt.figure(dpi=300)
count=0
for epsilon,mean_pol in a.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='epsilon='+str(epsilon),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('ε-greedy')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting 2/e-greedy'+temps+environment_names[0]+'.png')
plt.show()


#BEB NO PRIOR


betas=[0.5,1,2,3,4,5]
priors=[0.001]
b,conv_b=fitting_BEB(environment_names,betas=betas,priors=priors,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step,informative=False)
np.save('Parameter fitting 2/BEB no prior'+temps,b)
plt.figure(dpi=300)
count=0
for (beta,prior),mean_pol in b.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('BEB')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting 2/BEB no prior'+temps+'.png')
plt.show()

#BEB PRIOR

betas=[round(0.5*i,1) for i in range(1,11)]
priors=[round(0.5*i,1) for i in range(1,6)]
b,conv_b=fitting_BEB(environment_names,betas=betas,priors=priors,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/BEB'+temps,b)



markers_beta=markers[:len(betas)]
new_dict_BEB={prior:{} for prior in priors}
for (beta,prior), mean_pol in b.items():
    new_dict_BEB[prior][beta]=mean_pol

for prior in new_dict_BEB:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_BEB[prior].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers_beta[count])
        count+=1
    plt.title('BEB, prior='+str(prior))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/BEB'+str(prior)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_b.values()),
                  index=pd.MultiIndex.from_tuples(conv_b.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('Priors')
plt.ylabel('Betas')
plt.title('Pas de convergence BEB')
plt.savefig('Parameter fitting 2/BEB_heatmap'+temps+environment_names[0]+'.png')
plt.show()

#RALP

alphas=[round(0.5*i,1) for i in range(1,6)]
ms=[round(0.5*i,1) for i in range(1,10)]
c,conv_c=fitting_RALP(environment_names,alphas=alphas,ms=ms,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/RALP'+temps,c)


markers_m=markers[:len(ms)]
new_dict_RALP={alpha:{} for alpha in alphas}
for (alpha,m), mean_pol in c.items():
    new_dict_RALP[alpha][m]=mean_pol



for alpha in new_dict_RALP:
    plt.figure(dpi=300)
    count=0
    for m,mean_pol in new_dict_RALP[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),marker=markers_m[count])
        count+=1
    plt.title('RALP, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/RALP'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_c.values()),
                  index=pd.MultiIndex.from_tuples(conv_c.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('m')
plt.ylabel('alpha')
plt.title('Pas de convergence RmaxLP')
plt.savefig('Parameter fitting 2/RALP_heatmap'+environment_names[0]+temps+'.png')
plt.show()

plt.figure(dpi=300)
count=0
for (alpha,m),mean_pol in c.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('Rmax-LP α=0.5')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/RmaxLP'+temps+'.png')
plt.show()"""
"""
#BEBLP

alphas=[round(0.5*i,1) for i in range(1,6)]
betas=[round(0.5*i,1) for i in range(1,11)]

d,conv_d=fitting_BEBLP(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/BEBLP'+temps,d)


markers_beta=markers[:len(betas)]
new_dict_BEBLP={alpha:{} for alpha in alphas}
for (beta,alpha), mean_pol in d.items():
    new_dict_BEBLP[alpha][beta]=mean_pol



for alpha in new_dict_BEBLP:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_BEBLP[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers_beta[count])
        count+=1
    plt.title('BEBLP, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/BEBLP'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_d.values()),
                  index=pd.MultiIndex.from_tuples(conv_d.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Pas de convergence BEBLP')
plt.savefig('Parameter fitting 2/BEBLP_heatmap'+environment_names[0]+temps+'.png')
plt.show()


plt.figure(dpi=300)
count=0
for (beta,alpha),mean_pol in d.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('BEB-LP')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/BEBLP'+temps+'.png')
plt.show()
"""
#Rmax
"""
ms=[i for i in range(1,11)]
u_ms=[i for i in range(1,30,3)]
markers_m=markers[:len(ms)]
e,conv_e=fitting_RA(environment_names,u_ms=u_ms,ms=ms,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/Rmax'+temps,e)

new_dict={u_m:{} for u_m in u_ms}
for (m,u_m), mean_p in e.items():
    new_dict[u_m][m]=mean_p

for u_m in new_dict:
    plt.figure(dpi=300)
    count=0
    for m,mean_pol in new_dict[u_m].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),marker=markers_m[count])
        count+=1
    plt.title('Rmax, u_m='+str(u_m))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/Rmax'+str(u_m)+environment_names[0]+temps+'.png')
    plt.show()

ser = pd.Series(list(conv_e.values()),
                  index=pd.MultiIndex.from_tuples(conv_e.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('u_m')
plt.ylabel('m')
plt.title('Pas de convergence Rmax')
plt.savefig('Parameter fitting 2/Rmax_heatmap'+environment_names[0]+temps+'.png')
plt.show()



plt.figure(dpi=300)
count=0
for (m,u_m),mean_pol in e.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('Rmax')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/Rmax'+temps+'.png')
plt.show()
"""

"""
gamma_es=[0.1*i for i in range(6)]
coeff_es=[1*i for i in range(6)]
f,conv_f=fitting_EGE(environment_names,gamma_es=gamma_es,coeff_es=coeff_es,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting/EGE'+temps,f)

plt.figure(dpi=300)
for (gamma_e,coeff_e),mean_pol in f.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='gamma_e='+str(gamma_e)+', coeff_e='+str(coeff_e))
plt.title('EGE')
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/EGE'+environment_names[0]+temps+'.png')
plt.show()

ser = pd.Series(list(conv_f.values()),
                  index=pd.MultiIndex.from_tuples(conv_f.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('gamma_e')
plt.ylabel('coeff_e')
plt.title('Pas de convergence ε-greedy-E')
plt.savefig('Parameter fitting/EGE_heatmap'+environment_names[0]+temps+'.png')
plt.show()

# PIM    
    

alphas=[10**(i) for i in range(-3,4)]
betas=[10**(i) for i in range(-3,4)]
g,conv_g=fitting_PIM(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/PIM'+temps,g)


new_dict_PIM={alpha:{} for alpha in alphas}
for (alpha,beta), mean_p in g.items():
    new_dict_PIM[alpha][beta]=mean_p


for alpha in new_dict_PIM:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_PIM[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers[count])
        count+=1
    plt.title('PIM, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/PIM'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_g.values()),
                  index=pd.MultiIndex.from_tuples(conv_g.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('beta')
plt.ylabel('alpha')
plt.title('Pas de convergence PIM')
plt.savefig('Parameter fitting 2/PIM_heatmap'+environment_names[0]+temps+'.png')
plt.show()
"""

"""
alphas=[10**(i) for i in range(-3,4)]
betas=[10**(i) for i in range(-3,4)]
g,conv_g=fitting_PIM(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/PEI-DE'+temps,g)


new_dict_PIM={alpha:{} for alpha in alphas}
for (alpha,beta), mean_p in g.items():
    new_dict_PIM[alpha][beta]=mean_p


for alpha in new_dict_PIM:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_PIM[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers[count])
        count+=1
    plt.title('PIM, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/PEI-DE'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_g.values()),
                  index=pd.MultiIndex.from_tuples(conv_g.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('beta')
plt.ylabel('alpha')
plt.title('Pas de convergence PEI-DE')
plt.savefig('Parameter fitting 2/PEI-DE_heatmap'+environment_names[0]+temps+'.png')
plt.show()"""

