import numpy as np
import time
from main_Functions import play,loading_environments
from Lopesworld import Lopes_environment
import matplotlib.pyplot as plt
import seaborn as sns
import copy


from matplotlib.colors import TwoSlopeNorm

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
    environment=Lopes_environment(**environment_parameters[name_environment])
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
    opti_pol_errors,real_pol_errors={},{}
    for result in results : 
        opti_pol_errors[result[0]]=result[1][1]
        real_pol_errors[result[0]]=result[1][2]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    return opti_pol_errors,real_pol_errors


def get_mean_and_CI_fitting(dictionary,names_environments,agents_tested,number_of_iterations,first_hyperparameters,second_hyperparameters):
    mean={(name_agent,h_1,h_2): np.average([dictionary[name_agent,name_environment,i,h_1,h_2] for i in range(number_of_iterations) for name_environment in names_environments],axis=0)  
                                                 for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}    
    CI={(name_agent,h_1,h_2):np.std([dictionary[name_agent,name_environment,i,h_1,h_2] for name_environment in names_environments for i in range(number_of_iterations)],axis=0)/np.sqrt(number_of_iterations*len(names_environments)) 
                        for name_agent in agents_tested for h_1 in first_hyperparameters for h_2 in second_hyperparameters}
    return mean,CI

def extracting_results(opti_pol_error,real_pol_error,names_environments,agents_tested,number_of_iterations,first_hyperparameters,second_hyperparameters):
    mean_pol_opti,CI_pol_opti=get_mean_and_CI_fitting(opti_pol_error,names_environments,agents_tested,number_of_iterations,first_hyperparameters,second_hyperparameters)    
    mean_pol_real,CI_pol_real=get_mean_and_CI_fitting(real_pol_error,names_environments,agents_tested,number_of_iterations,first_hyperparameters,second_hyperparameters)
    return mean_pol_opti,CI_pol_opti,mean_pol_real,CI_pol_real


def get_best_performance(pol_error,name_agent,first_hyperparameters,second_hyperparameters,range_of_the_mean):
    avg_pol_error = {(name_agent,hp_1,hp_2):np.average(pol_error_value[-range_of_the_mean:]) for (name_agent,hp_1,hp_2),pol_error_value in pol_error.items()}
    array_result=np.zeros((len(first_hyperparameters)-1,len(second_hyperparameters)-1))
    for index_hp_1,hp_1 in enumerate(first_hyperparameters[1:]) :
        for index_hp_2,hp_2 in enumerate(second_hyperparameters[1:]) :
            array_result[(index_hp_1,index_hp_2)]=avg_pol_error[name_agent,hp_1,hp_2]
    return array_result

def save_results_parametter_fitting(pol_opti,CI_pol_opti,pol_real,CI_pol_real,name_agent,first_hyperparameters,second_hyperparameters,play_parameters,name_environments):
    time_end=str(round(time.time()%1e7))
    np.save('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_pol_opti_'+time_end+'.npy',pol_opti)
    np.save('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_CI_pol_opti_'+time_end+'.npy',CI_pol_opti)
    np.save('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_pol_real_'+time_end+'.npy',pol_real)
    np.save('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_CI_pol_real_'+time_end+'.npy',CI_pol_real)
    return time_end

def plot_from_saved(name_agent,first_hyperparameters,second_hyperparameters,step_between_VI,name_environments,time_end,opti):
    if opti :
        pol=np.load('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_pol_opti_'+time_end+'.npy',allow_pickle=True)[()]
        CI_pol=np.load('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_CI_pol_opti_'+time_end+'.npy',allow_pickle=True)[()]
    else : 
        pol=np.load('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_pol_real_'+time_end+'.npy',allow_pickle=True)[()]
        CI_pol=np.load('Parameter fitting/Data/'+name_agent+'_'+name_environments[0]+'_CI_pol_real_'+time_end+'.npy',allow_pickle=True)[()]
    plot_parameter_fitting(pol,CI_pol,name_agent,first_hyperparameters,second_hyperparameters,step_between_VI,name_environments,time_end,opti)

def plot_parameter_fitting(pol,CI_pol,name_agent,first_hyperparameters,second_hyperparameters,step_between_VI,name_environments,time_end,opti):
    
    rename={'RA':'R-max','BEB':'BEB','EBLP':'ζ-EB','RALP':'ζ-R-max','Epsilon_MB':'ε-greedy'}
    markers=['^','o','x','*','s']
    colors=['#9d02d7','#0000ff',"#ff7763","#ffac1e","#009435"]
    array_avg_pol_last_500=get_best_performance(pol,name_agent,first_hyperparameters,second_hyperparameters,10)
    
    plot_2D(array_avg_pol_last_500,first_hyperparameters,second_hyperparameters)
    if opti :
        plt.title(rename[name_agent]+' optimal policy - last 500 steps')
        plt.savefig('Parameter fitting/Heatmaps/heatmap_'+name_agent+' optimal_policy '+name_environments[0]+time_end+'.pdf')
    else : 
        plt.title(rename[name_agent]+' agent policy - last 500 steps')
        plt.savefig('Parameter fitting/Heatmaps/heatmap_'+name_agent+' real_policy '+name_environments[0]+time_end+'.pdf')
    plt.close() 
    
    curve_number=0
    for hp_1 in first_hyperparameters[1:] :
        basic_plot()
        for hp_2 in second_hyperparameters[1:] :
            yerr0 = pol[name_agent,hp_1,hp_2] - CI_pol[name_agent,hp_1,hp_2]
            yerr1 = pol[name_agent,hp_1,hp_2] + CI_pol[name_agent,hp_1,hp_2]

            plt.fill_between([step_between_VI*i for i in range(len(pol[name_agent,hp_1,hp_2]))], yerr0, yerr1, color=colors[curve_number], alpha=0.2)

            plt.plot([step_between_VI*i for i in range(len(pol[name_agent,hp_1,hp_2]))],pol[name_agent,hp_1,hp_2],color=colors[curve_number],
                     label=str(second_hyperparameters[0])+"="+str(hp_2),ms=4,marker=markers[curve_number])
            curve_number+=1
            if curve_number == 5 or hp_2 == second_hyperparameters[-1]: 
                plt.legend()
                if opti : 
                    plt.title(rename[name_agent]+" optimal policy with "+str(first_hyperparameters[0])+" = "+str(hp_1))
                    plt.savefig('Parameter fitting/1DPlots/pol_error_opti_'+name_agent+'_'+name_environments[0]+str(hp_2)+"_"+str(time.time())+'.pdf')
                else : 
                    plt.title(rename[name_agent]+" agent policy with "+str(first_hyperparameters[0])+" = "+str(hp_1))
                    plt.savefig('Parameter fitting/1DPlots/pol_error_real_'+name_agent+'_'+name_environments[0]+str(hp_2)+"_"+str(time.time())+'.pdf')
                plt.close()
                curve_number=0
                if hp_2 != second_hyperparameters[-1]: basic_plot()

def plot_2D(array_result,first_hyperparameters,second_hyperparameters):
    fig=plt.figure(dpi=300)
    fig.add_subplot(1, 1, 1)
    #rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
    divnorm = TwoSlopeNorm(vmin=-12, vcenter=-1, vmax=0)
    sns.heatmap(array_result, cmap='bwr', norm=divnorm,cbar=True,annot=np.round(array_result,1),
                cbar_kws={"ticks":[-12,-1,0]},annot_kws={"size": 35 / (np.sqrt(len(array_result))+2.5)})
    #linewidths=0.05, linecolor='black'
    plt.xlabel(second_hyperparameters[0])
    plt.ylabel(first_hyperparameters[0])
    plt.xticks([i+0.5 for i in range(len(second_hyperparameters[1:]))],second_hyperparameters[1:])
    plt.yticks([i+0.5 for i in range(len(first_hyperparameters[1:]))],first_hyperparameters[1:])
            

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

    opti_pol_errors,real_pol_errors=main_function(seeds_agent,every_simulation,play_parameters,agent_parameters)
    
    mean_pol_opti,CI_pol_opti,mean_pol_real,CI_pol_real=extracting_results(opti_pol_errors,real_pol_errors,environments,agent,nb_iters,first_hp[1:],second_hp[1:])
    
    time_end=save_results_parametter_fitting(mean_pol_opti,CI_pol_opti,mean_pol_real,CI_pol_real,agent_name,first_hp,second_hp,play_parameters,environments)
    
    for opti in [False,True]:plot_from_saved(agent_name,first_hp,second_hp,play_parameters["step_between_VI"],environments,time_end,opti)



from agents import Rmax_Agent, BEB_Agent, Epsilon_MB_Agent, EBLP_Agent, RmaxLP_Agent

# Parameter fitting for each agent #

environments_parameters=loading_environments()
play_params={'trials':100, 'max_step':30, 'screen':False,'photos':[10,20,50,80,99],'accuracy_VI':0.01,'step_between_VI':50}
environments=["Lopes"]
nb_iters=20


#Reproduction of Lopes et al. (2012)

#R-max
agent_RA={'RA':Rmax_Agent}
RA_basic_parameters={Rmax_Agent:{'gamma':0.95,'Rmax':1,'m':8,'m_uncertain_states':15,'condition':'informative'}}

#agents={'RA':Rmax_Agent,'RALP':RmaxLP_Agent,'BEB':BEB_Agent,'EBLP':EBLP_Agent,'Epsilon_MB':Epsilon_MB_Agent}


starting_seed=10000

first_hp_RA= ['m']+[i for i in range(5,41,5)]
second_hp_RA=['m_uncertain_states']+[i for i in range(5,41,5)]
fit_parameters_agent(environments,agent_RA,'RA',nb_iters,first_hp_RA,second_hp_RA,RA_basic_parameters,starting_seed,play_params)

first_hp_RA= ['m']+[i for i in range(2,15,2)]
second_hp_RA=['m_uncertain_states']+[i for i in range(2,21,2)]
fit_parameters_agent(environments,agent_RA,'RA',nb_iters,first_hp_RA,second_hp_RA,RA_basic_parameters,starting_seed,play_params)


#RALP

agent_RALP={'RALP':RmaxLP_Agent}
RALP_basic_parameters={RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'m':2,'alpha':0.3,'prior_LP':0.001}}
first_hp_RALP= ['m']+[round(i*0.1,1) for i in range(5,41,5)]
second_hp_RALP=['alpha']+[0.1]+[round(i*0.1,1) for i in range(5,41,5)]
starting_seed=20000

fit_parameters_agent(environments,agent_RALP,'RALP',nb_iters,first_hp_RALP,second_hp_RALP,RALP_basic_parameters,starting_seed,play_params)


#BEB 

agent_BEB={'BEB':BEB_Agent}
BEB_basic_parameters={BEB_Agent:{'gamma':0.95,'beta':3,'coeff_prior':0.001,'condition':'informative'}}

first_hp_BEB= ['coeff_prior']+[3]
second_hp_BEB=['beta']+[0.1]+[i for i in range(1,10)]
starting_seed=30000

fit_parameters_agent(environments,agent_BEB,'BEB',nb_iters,first_hp_BEB,second_hp_BEB,BEB_basic_parameters,starting_seed,play_params)

#EBLP

agent_EBLP={'EBLP':EBLP_Agent}
EBLP_basic_parameters={EBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'prior_LP':0.001,'alpha':0.4}}


first_hp_EBLP= ['beta']+[0.1]+[round(0.5*i,1) for i in range(1,8)]
second_hp_EBLP=['alpha']+[0.1]+[round(0.5*i,1) for i in range(1,5)]
starting_seed=40000

fit_parameters_agent(environments,agent_EBLP,'EBLP',nb_iters,first_hp_EBLP,second_hp_EBLP,EBLP_basic_parameters,starting_seed,play_params)


# Epsilon_MB

agent_Epsilon_MB={'Epsilon_MB':Epsilon_MB_Agent}
Epsilon_MB_basic_parameters={Epsilon_MB_Agent:{'gamma':0.95,'epsilon':0.2}}

first_hp_Epsilon_MB= ['gamma']+[0.95]
second_hp_Epsilon_MB=['epsilon']+[0.01]+[0.05]+[round(i*0.1,1) for i in range(1,4,1)]+[round(i*0.1,1) for i in range(4,11,2)]
starting_seed=50000

fit_parameters_agent(environments,agent_Epsilon_MB,'Epsilon_MB',nb_iters,first_hp_Epsilon_MB,second_hp_Epsilon_MB,Epsilon_MB_basic_parameters,starting_seed,play_params)

#Impact of the prior factor for BEB

agent_BEB={'BEB':BEB_Agent}
BEB_basic_parameters={BEB_Agent:{'gamma':0.95,'beta':3,'coeff_prior':0.001,'condition':'informative'}}

first_hp_BEB= ['beta']+[1]
second_hp_BEB=['coeff_prior']+[10**i for i in range(-1,4)]
starting_seed=60000

fit_parameters_agent(environments,agent_BEB,'BEB',nb_iters,first_hp_BEB,second_hp_BEB,BEB_basic_parameters,starting_seed,play_params)

#Parameter fitting without informative prior BEB

agent_BEB={'BEB':BEB_Agent}
BEB_basic_parameters={BEB_Agent:{'gamma':0.95,'beta':3,'coeff_prior':0.001,'condition':'uninformative'}}

first_hp_BEB= ['coeff_prior']+[0.001]
second_hp_BEB=['beta']+[0.1]+[i for i in range(1,10)]
starting_seed=70000

fit_parameters_agent(environments,agent_BEB,'BEB',nb_iters,first_hp_BEB,second_hp_BEB,BEB_basic_parameters,starting_seed,play_params)

#Parameter fitting without informative prior Rmax
starting_seed=80000

agent_RA={'RA':Rmax_Agent}
RA_basic_parameters={Rmax_Agent:{'gamma':0.95,'Rmax':1,'m':8,'m_uncertain_states':15,'condition':'uninformative'}}

first_hp_RA= ['gamma']+[0.95]
second_hp_RA=['m']+[i for i in range(2,21,2)]

fit_parameters_agent(environments,agent_RA,'RA',nb_iters,first_hp_RA,second_hp_RA,RA_basic_parameters,starting_seed,play_params)

#Impact of the learning progress prior
agent_EBLP={'EBLP':EBLP_Agent}
EBLP_basic_parameters={EBLP_Agent:{'gamma':0.95,'beta':2.4,'step_update':10,'prior_LP':0.001,'alpha':0.4}}


first_hp_EBLP= ['alpha']+[round(0.5*i,1) for i in range(1,11)]
second_hp_EBLP=['prior_LP']+[10**(i) for i in range(-3,2)]
starting_seed=90000

fit_parameters_agent(environments,agent_EBLP,'EBLP',nb_iters,first_hp_EBLP,second_hp_EBLP,EBLP_basic_parameters,starting_seed,play_params)


agent_RALP={'RALP':RmaxLP_Agent}
RALP_basic_parameters={RmaxLP_Agent:{'gamma':0.95,'Rmax':1,'step_update':10,'m':2,'alpha':0.3,'prior_LP':0.001}}
first_hp_RALP= ['gamma']+[0.95]
second_hp_RALP=['prior_LP']+[10**(i) for i in range(-5,4)]
starting_seed=100000

fit_parameters_agent(environments,agent_RALP,'RALP',nb_iters,first_hp_RALP,second_hp_RALP,RALP_basic_parameters,starting_seed,play_params)
