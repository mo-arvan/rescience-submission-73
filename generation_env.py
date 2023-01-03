import numpy as np

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4         

###Transition and reward generation functions ###

#Transitions
def transition_Lopes(alphas=[1,0.1]):
        transitions=np.zeros((25,5,25))
        uncertain_states=[1,3,11,13]
        
        for state in range(25):
            if state in uncertain_states:alpha=max(alphas)
            else : alpha=min(alphas)
            
            reachable_states_bool=np.array([state>4,state<20,state%5!=0,state%5!=4,True])
            result_of_the_action=np.array([-5,+5,-1,1,0])
            reachable_states=np.unique(state+reachable_states_bool*result_of_the_action)
            
            for action in range(5):
                    
                values=np.random.dirichlet([alpha]*len(reachable_states))
                deterministic_state = state+reachable_states_bool[action]*result_of_the_action[action]
                
                for index_arrival_state,arrival_state in enumerate(reachable_states):
                    transitions[state,action,arrival_state]=values[index_arrival_state]
                    
                if transitions[state,action,deterministic_state]!=np.max(values) :
                    state_max_value=np.argmax(transitions[state,action])
                    tSA=transitions[state,action]
                    tSA[deterministic_state],tSA[state_max_value]=tSA[state_max_value],tSA[deterministic_state]
                    
        return transitions

#Non-stationarity in the article
                                                                   
states_optimal_path=[0,5,15,16,17,18,19,14]

def non_stat_Lopes_article(transitions,index_state_to_change):
        copy_transitions=np.copy(transitions)
        state_to_change=states_optimal_path[index_state_to_change]
        
        
        derangement_list=np.arange(5)
        while np.any(derangement_list == np.arange(5)):
            derangement_list=np.random.permutation(derangement_list)
        

        for action in range(5):
            copy_transitions[state_to_change][derangement_list[action]]=transitions[state_to_change][action]
        return copy_transitions

#Non-stationarity on all the states of the optimal path

def non_stat_Lopes_all_states(transitions):
    for index in range(len(states_optimal_path)):
        transitions=non_stat_Lopes_article(transitions,index_state_to_change=index)
    return transitions

#Reward generation

def reward_Lopes(bonus=1,malus=-0.1):
    array_rewards = np.zeros(25)
    lopes_rewards={6: malus, 7: malus, 8 : malus, 12 : malus, 14: bonus}
    for state,reward in lopes_rewards.items():
        array_rewards[state]=reward
    return array_rewards

def save_rewards(bonus=1,malus=-0.1):
    array_rewards=reward_Lopes(bonus,malus)
    np.save('Environments/Rewards_Lopes_'+str(bonus)+'_'+str(malus)+'.npy',array_rewards)
    

#Validity of the worlds generated     

from policy_Functions2 import value_iteration
from Lopesworld2 import Lopes_environment

#Use only worlds in which the optimal path corresponds to the one in the article    
def valid_policy(policy):
    return (policy[0]==1 and policy[5]==1 and policy[10]==1 and policy[15]==3 and
            policy[16]==3 and policy[17]==3 and policy[18]==3 and policy[19]==0)

def valid_Lopes(alphas=[1,0.1],malus=-0.1,bonus=1):
        transitions,rewards=transition_Lopes(alphas),reward_Lopes(malus=malus,bonus=bonus)
        environment=Lopes_environment(transitions,rewards)
        _,policy=value_iteration(environment,0.95,0.01)
        return valid_policy(policy),transitions

#Statistics about the proportion of valid worlds           
def proportion_of_valid_worlds(iterations=2000,alphas=[1,0.1],malus=-0.1,bonus=1):
    valid_count=0
    for i in range(iterations):
        validity,_=valid_Lopes(alphas,malus,bonus)
        valid_count+=validity
    print('')    
    print('Percentage of valid worlds out of '+str(iterations)+' sampled worlds, with a malus of '+str(malus)+': '+str(round(100*valid_count/iterations,1))+'%')

#Functions to generate each world
def generate_valid_stationary_environments(number_of_worlds=1,alphas=[1,0.1],malus=-0.1,bonus=1): 
    for index_world in range(1,number_of_worlds+1):
        counter,validity=0,False
        while not validity and counter < 1000 : 
            validity,transitions=valid_Lopes(alphas,malus,bonus)
            counter+=1
        if counter ==1000 : raise RuntimeError('No world found with the optimal policy')
        np.save('Environments/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',transitions)


def generate_non_stationarity_article(world_number=1,number_of_worlds=1,malus=-0.1):
    for index_world in range(1,world_number+1):
        transitions=np.load('Environments/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',allow_pickle=True)
        for non_stat_number in range(1,number_of_worlds+1):
            transitions_non_stationary_article=non_stat_Lopes_article(transitions,np.random.randint(len(states_optimal_path)))
            np.save('Environments/Transitions_non_stat_article'+str(malus)+'_'+str(index_world)+'_'+str(non_stat_number)+'.npy',transitions_non_stationary_article)
        
def generate_strong_non_stationarity_(world_number=1,number_of_worlds=1,malus=-0.1):
    for index_world in range(1,world_number+1):
        transitions=np.load('Environments/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',allow_pickle=True)
        for non_stat_number in range(1,number_of_worlds+1):
            transitions_strong_non_stationarity=non_stat_Lopes_all_states(transitions)
            np.save('Environments/Transitions_strong_non_stat_'+str(malus)+'_'+str(index_world)+'_'+str(non_stat_number)+'.npy',transitions_strong_non_stationarity)



#generate the reward matrices 
save_rewards(1,-0.1)
save_rewards(1,-1)

#Generating a world similar to the one in the article
np.random.seed(1)
generate_valid_stationary_environments(number_of_worlds=1,alphas=[1,0.1],malus=-0.1,bonus=1)

#Generating 20 worlds with one change in the optimal path
np.random.seed(2)
generate_non_stationarity_article(world_number=1,number_of_worlds=20,malus=-0.1)

#Generating 20 changes in the whole optimal path
np.random.seed(3)
generate_strong_non_stationarity_(world_number=1,number_of_worlds=20,malus=-0.1)

#Generating 10 worlds with Bureau & Sebag (2013) parameters
np.random.seed(4)
generate_valid_stationary_environments(number_of_worlds=10,alphas=[1,0.1],malus=-1,bonus=1)

#Generating 10 changes in the optimal path for each world
np.random.seed(5)
generate_non_stationarity_article(world_number=10,number_of_worlds=10,malus=-1)

#Generating 10 changes in the whole optimal path
np.random.seed(6)
generate_strong_non_stationarity_(world_number=10,number_of_worlds=10,malus=-1)

#Checking how many worlds are valid out of 1000 for each condition
np.random.seed(7)
proportion_of_valid_worlds(iterations=1000,alphas=[1,0.1],malus=-0.1,bonus=1)
proportion_of_valid_worlds(iterations=1000,alphas=[1,0.1],malus=-3,bonus=1)