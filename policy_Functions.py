import numpy as np

def value_iteration(environment,gamma,accuracy):
    V={state:1 for state in environment.states}
    policy={state:0 for state in environment.states}
    delta=accuracy+1
    while delta > accuracy :
        delta=0
        for state,value in V.items():
            value_V=V[state]
            V[state]=np.max([np.sum([environment.transitions[action][state][new_state]*(environment.values[new_state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
            delta=max(delta,np.abs(value_V-V[state]))
    for state,value in V.items():
        policy[state]=np.argmax([np.sum([environment.transitions[action][state][new_state]*(environment.values[new_state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action in environment.actions])
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
            V[state]=np.sum([probability_action*np.sum([environment.transitions[action][state][new_state]*(environment.values[new_state]+gamma*V[new_state]) for new_state in environment.transitions[action][state].keys()]) for action,probability_action in actions.items()])
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

def get_agent_current_policy(agent,environment,gamma,accuracy):
    action_every_state=dict()
    for state in agent.Q.keys():
        q_values = agent.Q[state]
        max_value_state=max(q_values.values())
        best_action = np.random.choice([k for k, v in q_values.items() if v == max_value_state])
        action_every_state[state]={best_action:1}
        if type(agent).__name__=='Epsilon_MB_Agent':
            random_actions=list(range(5))
            del random_actions[best_action]
            action_every_state[state]={other_action: agent.epsilon/5 for other_action in random_actions}
            action_every_state[state][best_action]=1-agent.epsilon+agent.epsilon/5
    return action_every_state

    


