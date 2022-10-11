import numpy as np
UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

#If the agent can move from one state to all the others
def transition_Lopes_all_states():
        dict_transitions=[[list({} for i in range(5))for i in range(5)] for i in range(5)]
        uncertain_states=[(0,1),(0,3),(2,1),(2,3)]
        for action in [UP,DOWN,LEFT,RIGHT,STAY]:
            for height in range(5):
                for width in range(5):
                    if (height,width) not in uncertain_states:probas=np.random.dirichlet([0.1]*25)
                    else : probas = np.random.dirichlet([1]*25)
                    probas=probas.reshape(5,5)
                    if action == UP and height-1 >=0: index=(height-1,width)
                    elif action == DOWN and height+1 <5: index = (height+1,width)
                    elif action == LEFT and width-1 >=0: index = (height,width-1)
                    elif action == RIGHT and width+1 <5: index = (height,width+1)
                    else : index=(height,width) 
                    max_index=np.unravel_index(probas.argmax(),probas.shape)
                    probas[max_index],probas[index]=probas[index],probas[max_index]
                    for row in range(5):
                        for col in range(5):
                            dict_transitions[action][height][width][(row,col)]=probas[row][col]
        return dict_transitions
    
#Generation of the transitions in a slightly different way 
def transition_Lopes_2():
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
                    if height-1 <0: etats.remove((height-1,width))
                    if height+1 ==5: etats.remove((height+1,width))
                    if width-1 <0:  etats.remove((height,width-1))
                    if width+1==5:  etats.remove((height,width+1))
                    values=np.random.dirichlet([alpha]*len(etats))
                    probas={etats[i]:values[i] for i in range(len(etats))}
                    maxValue = max(probas.values())
                    max_ind = [k for k, v in probas.items() if v == maxValue]
                    j=np.random.randint(len(max_ind))                   
                    probas[max_ind[j]],probas[ind]=probas[ind],probas[max_ind[j]]
                    for key in probas.keys() :
                            dict_transitions[action][height][width][key]=probas[key]
        return dict_transitions