import numpy as np 


from Lopesworld import Lopes_State

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4         
#Generate transitions of the worlds used in the article

def transition_Lopes():
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) not in uncertain_states:alpha=0.1
                    else : alpha=1
                    if action == UP and height-1 >=0: ind=(height-1,width)
                    elif action == DOWN and height+1 <5: ind = (height+1,width)
                    elif action == LEFT and width-1 >=0: ind = (height,width-1)
                    elif action == RIGHT and width+1 <5: ind = (height,width+1)
                    else : ind=(height,width) 
                    etats=[(height,width), (height-1,width),(height+1,width),(height,width-1),(height,width+1)]
                    values=np.random.dirichlet([alpha]*5)
                    values=np.random.dirichlet([alpha]*len(etats))
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

#Generate one non_stationary world for each stationary world 

def non_stat_Lopes_article(world_number=1):
        transitions=np.load('Mondes/Transitions_Lopes_'+str(world_number)+'.npy',allow_pickle=True)
        optimal_path=[(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4)]
        
        index_changed=np.random.randint(len(optimal_path))
        state_to_change=optimal_path[index_changed]
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
        np.save('Mondes/Transitions_Lopes_no_stat '+str(world_number)+'.npy',transitions)
          
        
from policy_Functions import value_iteration

#Use only worlds in which the optimal path corresponds to the one in the article    
def valid_Lopes(malus=-0.1):
    for count in range(200):
        transitions=np.array(transition_Lopes())
        environment=Lopes_State(transitions,malus)
        _,policy=value_iteration(environment,0.95,0.01)
        if policy[0,0]==1 and policy[1,0]==1 and policy[2,0]==1 and policy[3,0]==3 and policy[3,1]==3 and policy[3,2]==3 and policy[3,3]==3 :
            return transitions
            
def proportion_valid_Lopes(malus=-0.1):
    count=0
    iteration=1000
    for i in range(iteration):
        if i%100==0: print(i)
        transitions=np.array(transition_Lopes())
        environment=Lopes_State(transitions,malus)
        _,policy=value_iteration(environment,0.95,0.01)
        if policy[0,0]==1 and policy[1,0]==1 and policy[2,0]==1 and policy[3,0]==3 and policy[3,1]==3 and policy[3,2]==3 and policy[3,3]==3 and policy[3,4]==0 :
            count+=1
    print(count/iteration)

def generer_20_Lopes_valid(malus): 
    for i in range(1,21):
        transitions=valid_Lopes(malus)
        np.save('Mondes/Transitions_Lopes_'+str(i)+'.npy',transitions)
        non_stat_Lopes_article(i)
        
        