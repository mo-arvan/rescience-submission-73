import numpy as np

def value_iteration(environment,gamma,accuracy):
    V={state:1 for state in environment.states}
    policy={state:0 for state in environment.states}
    delta=accuracy+1
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            V[state]=np.max([np.sum([environment.transitions[action][state][new_state]*(environment.values[state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
            delta=max(delta,np.abs(value_V-V[state]))
    for state,value in V.items():
        policy[state]=np.argmax([np.sum([environment.transitions[action][state][new_state]*(environment.values[state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
    return V,policy


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
            V[state]=np.sum([probability_action*np.sum([environment.transitions[action][state][new_state]*(environment.values[state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action,probability_action in actions.items()])
            delta=max(delta,np.abs(value_V-V[state]))
    return V

def get_optimal_policy(agent,environment,gamma,accuracy):
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





