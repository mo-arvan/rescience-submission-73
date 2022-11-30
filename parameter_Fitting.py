import numpy as np
import time
from Useful_functions import play,loading_environments,save_and_plot
from Lopesworld import Lopes_State
import matplotlib.pyplot as plt
### Parametter fitting ##

from multiprocessing import Pool
example = ['beta']+[i for i in range(12)]
example2 =['coeff_prior']+[j for j in range(22,33)]

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
        d=basic_parameters.copy()
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
    return {(name_agent,hp_1,hp_2):np.average(pol_error_value[:-20]) for (name_agent,hp_1,hp_2),pol_error_value in pol_error_parameter.items()}

def plot_parametter_fitting(pol,CI_pol,reward,CI_reward,name_agent,first_hyperparameters,second_hyperparameters,play_parameters,name_environments):
    time_end=str(round(time.time()%1e7))
    markers=['^','o','x','*','s','P','.','D','1','v',',']
    colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    np.save('Results/'+time_end+name_environments[0]+'_polerror.npy',pol)
    rename={'RA':'R-max','BEB':'BEB','BEBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'Ɛ-greedy'}
    fig=plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    count=0
    for hp_1 in first_hyperparameters[1:] :
        for hp_2 in second_hyperparameters[1:] :
            yerr0 = pol[name_agent,hp_1,hp_2] - CI_pol[name_agent,hp_1,hp_2]
            yerr1 = pol[name_agent,hp_1,hp_2] + CI_pol[name_agent,hp_1,hp_2]

            plt.fill_between([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent,hp_1,hp_2]))], yerr0, yerr1, color=colors[count], alpha=0.2)

            plt.plot([play_parameters['step_between_VI']*i for i in range(len(pol[name_agent,hp_1,hp_2]))],pol[name_agent,hp_1,hp_2],color=colors[count],
                     label=str(second_hyperparameters[0])+"="+str(hp_2),ms=3,marker=markers[count])
            count+=1
        count=0
        plt.xlabel("Steps")
        plt.ylabel("Policy value error")
        plt.title(rename[name_agent]+" with "+str(first_hyperparameters[0])+" = "+str(hp_1))
        plt.grid(linestyle='--')
        plt.ylim([-12.5,0.5])
        plt.legend()
        plt.savefig('Results/pol_error'+name_agent+name_environments[0]+time_end+'.png')
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





from Rmax import Rmax_Agent
from BEB import BEB_Agent
from greedyMB import Epsilon_MB_Agent
from BEBLP import BEBLP_Agent
from RmaxLP import RmaxLP_Agent




environments_parameters=loading_environments()


#First experiment

RA_basic_parameters={Rmax_Agent:{'gamma':0.95, 'm':8,'Rmax':1,'u_m':15,'correct_prior':True}}

#environments=["Non_stat_article_-1_{0}".format(world)+'_{0}'.format(non_stat) for world in range(1,11) for non_stat in range(1,11)]
environments=["Lopes"]
#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'BEBLP':BEBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}
agent={'RA':Rmax_Agent}
nb_iters=1

play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}



example = ['m']+[1,3,5,10]
example2 =['u_m']+[1,3,5,10]

every_simulation=getting_simulations_to_do(environments,agent,nb_iters,example,example2)
all_seeds=[i for i in range(len(every_simulation))]
agent_parameters=nb_iters*get_agent_parameters(RA_basic_parameters,agent['RA'],example, example2)


pol_errors,rewards=main_function(all_seeds,every_simulation,play_params,agent_parameters)
pol,CI_pol, reward, CI_reward=extracting_results(rewards,pol_errors,environments,agent,nb_iters,example[1:],example2[1:])
plot_parametter_fitting(pol,CI_pol,reward,CI_reward,'RA',example,example2,play_params,environments)



