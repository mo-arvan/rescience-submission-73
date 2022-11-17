import numpy as np

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4         

###Transition and reward generation functions ###

#Transitions
def transition_Lopes(alphas=[1,0.1]):
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) in uncertain_states:alpha=max(alphas)
                    else : alpha=min(alphas)
                    if action == UP and height-1 >=0: ind=(height-1,width)
                    elif action == DOWN and height+1 <5: ind = (height+1,width)
                    elif action == LEFT and width-1 >=0: ind = (height,width-1)
                    elif action == RIGHT and width+1 <5: ind = (height,width+1)
                    else : ind=(height,width) 
                    etats=[(height,width), (height-1,width),(height+1,width),(height,width-1),(height,width+1)]
                    values=np.random.dirichlet([alpha]*5)
                    probas={etats[i]:values[i] for i in range(len(etats))}
                    maxValue = max(probas.values())
                    max_ind = [k for k, v in probas.items() if v == maxValue]
                    j=np.random.randint(len(max_ind))                   
                    probas[max_ind[j]],probas[ind]=probas[ind],probas[max_ind[j]]
                    if height-1 <0: 
                        probas[(height,width)]+=probas[(height-1,width)]
                        del probas[(height-1,width)]
                    if height+1 ==5:
                        probas[(height,width)]+=probas[(height+1,width)]
                        del probas[(height+1,width)]
                    if width-1 <0:
                        probas[(height,width)]+=probas[(height,width-1)]
                        del probas[(height,width-1)]
                    if width+1==5:
                        probas[(height,width)]+=probas[(height,width+1)]
                        del probas[(height,width+1)]
                    for key in probas.keys() :
                            dict_transitions[action][height][width][key]=probas[key]
        return dict_transitions

#Non-stationarity in the article
                                                                   
states_optimal_path=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4)]

def non_stat_Lopes_article(transitions,index_state_to_change=np.random.randint(len(states_optimal_path))):
        state_to_change=states_optimal_path[index_state_to_change]
        liste_rotation=[j for j in range(5)]
        valid=False
        while not valid:
            valid=True
            for k in range(5):
                if liste_rotation[k]==k:
                    valid=False
                    np.random.shuffle(liste_rotation)
                    break
        new_transitions=[transitions[rotation][state_to_change[0]][state_to_change[1]] for rotation in liste_rotation]
        for action in range(5):
            transitions[action][state_to_change[0]][state_to_change[1]]=new_transitions[action]
        return transitions

#Non-stationarity on all the states of the optimal path

def non_stat_Lopes_all_states(transitions):
    for index in range(len(states_optimal_path)):
        transitions=non_stat_Lopes_article(transitions,index_state_to_change=index)
    return transitions

#Reward generation

def reward_Lopes(bonus=1,malus=-0.1):
    array_rewards = np.zeros((5,5))
    lopes_rewards={(1,1):malus, (1,2): malus, (1,3): malus, (2,2) : malus, (2,4): bonus}
    for state,reward in lopes_rewards.items():
        array_rewards[state]=reward
    return array_rewards

def save_rewards(bonus=1,malus=-0.1):
    array_rewards=reward_Lopes(bonus,malus)
    np.save('Mondes/Rewards_Lopes_'+str(bonus)+'_'+str(malus)+'.npy',array_rewards)
        
#Validity of the worlds generated     

from policy_Functions import value_iteration
from Lopesworld import Lopes_State

#Use only worlds in which the optimal path corresponds to the one in the article    
def valid_policy(policy):
    return (policy[0,0]==1 and policy[1,0]==1 and policy[2,0]==1 and policy[3,0]==3 and
            policy[3,1]==3 and policy[3,2]==3 and policy[3,3]==3 and policy[3,4]==0)

def valid_Lopes(alphas=[1,0.1],malus=-0.1,bonus=1):
        transitions,rewards=np.array(transition_Lopes(alphas)),reward_Lopes(malus=malus)
        environment=Lopes_State(transitions,rewards)
        _,policy=value_iteration(environment,0.95,0.01)
        return valid_policy(policy),transitions

#Statistics about the proportion of valid worlds           
def proportion_valid_Lopes(iterations=1000,alphas=[1,0.1],malus=-0.1,bonus=1):
    valid_count=0
    for i in range(iterations):
        if i%100==0: print('Loading '+str(int(100*i/iterations))+'%')
        validity,_=valid_Lopes(alphas,malus,bonus)
        valid_count+=validity
    print('')    
    print('Percentage of valid worlds out of '+str(iterations)+' sampled worlds: ' +str(round(100*valid_count/iterations,1))+'%')

#Functions to generate each world
def generate_valid_stationary_environments(number_of_worlds=1,alphas=[1,0.1],malus=-0.1,bonus=1): 
    for index_world in range(1,number_of_worlds+1):
        counter,validity=0,False
        while not validity and counter < 1000 : 
            validity,transitions=valid_Lopes(alphas,malus,bonus)
            counter+=1
        if counter ==1000 : raise RuntimeError('A world does not have the optimal policy')
        np.save('Mondes/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',transitions)


def generate_non_stationarity_article(world_number=1,number_of_worlds=1,malus=-0.1):
    for index_world in range(1,world_number+1):
        transitions=np.load('Mondes/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',allow_pickle=True)
        for non_stat_number in range(1,number_of_worlds+1):
            transitions_non_stationary_article=non_stat_Lopes_article(transitions)
            np.save('Mondes/Transitions_non_stat_article'+str(malus)+'_'+str(index_world)+'_'+str(non_stat_number)+'.npy',transitions_non_stationary_article)
        
def generate_strong_non_stationarity_(world_number=1,number_of_worlds=1,malus=-0.1):
    for index_world in range(1,world_number+1):
        transitions=np.load('Mondes/Transitions_Lopes_'+str(malus)+'_'+str(index_world)+'.npy',allow_pickle=True)
        for non_stat_number in range(1,number_of_worlds+1):
            transitions_strong_non_stationarity=non_stat_Lopes_all_states(transitions)
            np.save('Mondes/Transitions_strong_non_stat_'+str(malus)+'_'+str(index_world)+'_'+str(non_stat_number)+'.npy',transitions_strong_non_stationarity)



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

