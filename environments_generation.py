import numpy as np

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4         

#Transition_generation
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
#Reward generation

def reward_Lopes(bonus=1,malus=-0.1):
    array_rewards = np.zeros((5,5))
    lopes_rewards={(1,1):malus, (1,2): malus, (1,3): malus, (2,2) : malus, (2,4): bonus}
    for state,reward in lopes_rewards.items():
        array_rewards[state]=reward
    return array_rewards

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
        
        
