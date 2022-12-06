import numpy as np
import time
from Useful_functions import play,loading_environments
from Lopesworld import Lopes_State
import matplotlib.pyplot as plt
import seaborn as sns
import copy


### Parameter fitting ##

from multiprocessing import Pool


def range_parameters_agent(list1,list2):
    list_of_params=[]
    for elem1 in range(1,len(list1)) :
        for elem2 in range(1,len(list2)):
            list_of_params.append({list1[0]:list1[elem1],list2[0]:list2[elem2]})
    return list_of_params

def getting_simulations_to_do(names_environments,agents_tested,number_of_iterations,values_first_hyperparameter,values_second_hyperparameter):
    every_simulation=[]
    for name_environment in names_environments:   
        for name_agent,agent in agents_tested.items(): 
            for iteration in range(number_of_iterations):
                for hyperparameter_1 in values_first_hyperparameter[1:] : 
                    for hyperparameter_2 in values_second_hyperparameter[1:] :
                        every_simulation.append((name_environment,name_agent,agent,iteration,hyperparameter_1,hyperparameter_2))
    return every_simulation

def get_agent_parameters(basic_parameters,agent,list_1,list_2):
    agent_parameters=[]
    list_of_new_parameters=range_parameters_agent(list_1, list_2)
    for dic in list_of_new_parameters:
        d=copy.deepcopy(basic_parameters)
        for key,value in dic.items() : 
            d[agent][key]=value
        agent_parameters.append(d)
    return agent_parameters

environment_parameters=loading_environments()

def play_with_params(name_environment,name_agent,agent,iteration,first_hyperparam,second_hyperparam,play_parameters,seed,agent_parameters):
    np.random.seed(seed)
    environment=Lopes_State(**environment_parameters[name_environment])
    globals()[name_agent]=agent(environment,**agent_parameters[agent])
    return (name_agent,name_environment,iteration,first_hyperparam,second_hyperparam),play(environment,globals()[name_agent],**play_parameters)

def one_parameter_play_function(args):
    return play_with_params(args[0][0],args[0][1],args[0][2],args[0][3],args[0][4],args[0][5],args[1],args[2],args[3])

def main_function(all_seeds,every_simulation,play_params,agent_parameters) :
    before=time.time()
    all_parameters=[[every_simulation[seed],play_params,all_seeds[seed], agent_parameters[seed]] for seed in range(len(all_seeds))]
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


def extracting_results(rewards,pol_error,names_environments,agents_tested,number_of_iterations,first_hyperparameters,second_hyperparameters):
    mean_pol_error_agent={(name_agent,h_1,h_2): np.average([pol_error[name_agent,name_environment,i,h_1,h_2] for i in range(number_of_iterations) for name_environment in names_environments],axis=0)  
                                                 for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}    
    CI_pol_error_agent={(name_agent,h_1,h_2):1.96*np.std([pol_error[name_agent,name_environment,i,h_1,h_2] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) 
                        for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}
    rewards_agent={(name_agent,h_1,h_2): np.average([rewards[name_agent,name_environment,i,h_1,h_2] for i in range(number_of_iterations) for name_environment in names_environments],axis=0)  
                                                 for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}
    CI_rewards_agent={(name_agent,h_1,h_2):1.96*np.std([rewards[name_agent,name_environment,i,h_1,h_2] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) 
                        for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}
    return mean_pol_error_agent,CI_pol_error_agent, rewards_agent, CI_rewards_agent


def get_best_performance(pol_error_parameter):
    return {(name_agent,hp_1,hp_2):np.average(pol_error_value[-10:]) for (name_agent,hp_1,hp_2),pol_error_value in pol_error_parameter.items()}

def plot_parameter_fitting(pol,CI_pol,reward,CI_reward,name_agent,first_hyperparameters,second_hyperparameters,play_parameters,name_environments):
    time_end=str(round(time.time()%1e7))
    np.save('Parameter fitting/'+time_end+'_'+name_environments[0]+'_'+name_agent+'_polerror.npy',pol)
    rename={'RA':'R-max','BEB':'BEB','BEBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'Ɛ-greedy'}
    
    
    avg_pol_last_1000_steps=get_best_performance(pol)
    array_result=np.zeros((len(first_hyperparameters)-1,len(second_hyperparameters)-1))
    for index_hp_1,hp_1 in enumerate(first_hyperparameters[1:]) :
        for index_hp_2,hp_2 in enumerate(second_hyperparameters[1:]) :
            array_result[(index_hp_1,index_hp_2)]=avg_pol_last_1000_steps[name_agent,hp_1,hp_2]
    
    array_result[array_result < max(np.median(array_result),-1)] = max(np.median(array_result),-1)
    
    fig=plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    sns.heatmap(array_result,cmap='bwr')
    plt.xlabel(second_hyperparameters[0])
    plt.ylabel(first_hyperparameters[0])
    plt.xticks([i+0.5 for i in range(len(second_hyperparameters[1:]))],second_hyperparameters[1:])
    plt.yticks([i+0.5 for i in range(len(first_hyperparameters[1:]))],first_hyperparameters[1:])
    plt.title('Mean policy value error on the last 500 steps')
    plt.savefig('Parameter fitting/heatmap'+name_agent+name_environments[0]+time_end+'.png')
    plt.close()
    
    markers=['^','o','x','*','s']
    colors=['#9d02d7','#0000ff',"#ff7763","#ffac1e","#009435"]
    count=0
    for hp_1 in first_hyperparameters[1:] :
        basic_plot()
        for hp_2 in second_hyperparameters[1:] :
            yerr0 = pol[name_agent,hp_1,hp_2] - CI_pol[name_agent,hp_1,hp_2]
            yerr1 = pol[name_agent,hp_1,hp_2] + CI_pol[name_agent,hp_1,hp_2]

            plt.fill_between([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent,hp_1,hp_2]))], yerr0, yerr1, color=colors[count], alpha=0.2)

            plt.plot([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent,hp_1,hp_2]))],pol[name_agent,hp_1,hp_2],color=colors[count],
                     label=str(second_hyperparameters[0])+"="+str(hp_2),ms=4,marker=markers[count])
            count+=1
            if count == 5 or hp_2 == second_hyperparameters[-1]: 
                plt.title(rename[name_agent]+" with "+str(first_hyperparameters[0])+" = "+str(hp_1))
                plt.legend()
                plt.savefig('Parameter fitting/pol_error'+name_agent+name_environments[0]+str(hp_2)+"_"+str(time.time())+'.png')
                plt.close()
                count=0
                if hp_2 != second_hyperparameters[-1]: basic_plot()
                
def basic_plot():
        fig=plt.figure(dpi=300)
        fig.add_subplot(1, 1, 1)
        plt.xlabel("Steps")
        plt.ylabel("Policy value error")
        plt.grid(linestyle='--')
        plt.ylim([-12.5,0.5])

def fit_parameters_agent(environments,agent,agent_name,nb_iters,first_hp,second_hp,agent_basic_parameters,starting_seed,play_parameters):
    
    every_simulation=getting_simulations_to_do(environments,agent,nb_iters,first_hp,second_hp)
    seeds_agent=[starting_seed+i for i in range(len(every_simulation))]
    agent_parameters=nb_iters*get_agent_parameters(agent_basic_parameters,agent[agent_name],first_hp, second_hp)

    pol_errors,rewards=main_function(seeds_agent,every_simulation,play_parameters,agent_parameters)
    pol,CI_pol, reward, CI_reward=extracting_results(rewards,pol_errors,environments,agent,nb_iters,first_hp[1:],second_hp[1:])
    plot_parameter_fitting(pol,CI_pol,reward,CI_reward,agent_name,first_hp,second_hp,play_parameters,environments)



from agents import Rmax_Agent, BEB_Agent, Epsilon_MB_Agent, BEBLP_Agent, RmaxLP_Agent

# Parameter fitting for each agent #

environments_parameters=loading_environments()
play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}
environments=["Lopes"]
nb_iters=20


#Reproduction of Lopes et al. (2012)

#R-max
agent_RA={'RA':Rmax_Agent}
RA_basic_parameters={Rmax_Agent:{'gamma':0.95,'Rmax':1,'m':8,'m_uncertain_states':15,'correct_prior':True}}

#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}


first_hp_RA= ['m']+[i for i in range(3,11,1)]
second_hp_RA=['m_uncertain_states']+[i for i in range(5,17,1)]
starting_seed=10000
fit_parameters_agent(environments,agent_RA,'RA',nb_iters,first_hp_RA,second_hp_RA,RA_basic_parameters,starting_seed,play_params)

#RALP

agent_RALP={'RALP':RmaxLP_Agent}
RALP_basic_parameters={RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'m':2,'alpha':0.3,'prior_LP':0.001}}
first_hp_RALP= ['m']+[round(i*0.1,1) for i in range(1,31,3)]
second_hp_RALP=['alpha']+[round(i*0.1,1) for i in range(1,21,2)]
starting_seed=20000

fit_parameters_agent(environments,agent_RALP,'RALP',nb_iters,first_hp_RALP,second_hp_RALP,RALP_basic_parameters,starting_seed,play_params)

#BEB 

agent_BEB={'BEB':BEB_Agent}
BEB_basic_parameters={BEB_Agent:{'gamma':0.95,'beta':3,'coeff_prior':0.001,'correct_prior':True,'informative':True}}

first_hp_BEB= ['beta']+[round(i*0.1,1) for i in range(25,76,5)]
second_hp_BEB=['coeff_prior']+[round(i*0.1,1) for i in range(1,31,5)]
starting_seed=30000

fit_parameters_agent(environments,agent_BEB,'BEB',nb_iters,first_hp_BEB,second_hp_BEB,BEB_basic_parameters,starting_seed,play_params)

#BEBLP

agent_BEBLP={'BEBLP':BEBLP_Agent}
BEBLP_basic_parameters={BEBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'prior_LP':0.001,'alpha':0.4}}


first_hp_BEBLP= ['beta']+[round(i*0.1,1) for i in range(15,71,5)]
second_hp_BEBLP=['alpha']+[round(i*0.1,1) for i in range(1,14,2)]
starting_seed=40000

fit_parameters_agent(environments,agent_BEBLP,'BEBLP',nb_iters,first_hp_BEBLP,second_hp_BEBLP,BEBLP_basic_parameters,starting_seed,play_params)


