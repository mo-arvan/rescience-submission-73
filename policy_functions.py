import numpy as np


def value_iteration(environment, gamma, accuracy):
    Q = np.zeros((len(environment.states), len(environment.actions)))
    V = np.zeros((len(environment.states)))
    converged = False
    while not converged:
        Q = np.dot(environment.transitions, environment.rewards+gamma*V)
        new_V = np.max(Q, axis=1)
        diff = np.abs(V - new_V)
        V = new_V
        if np.max(diff) < accuracy:
            converged = True
    policy = np.argmax(Q, axis=1)
    return V, policy


def policy_evaluation(environment, policy, gamma, accuracy):
    Q = np.zeros((len(environment.states), len(environment.actions)))
    V = np.zeros((len(environment.states)))
    converged = False
    while not converged:

        Q = np.dot(environment.transitions, environment.rewards+gamma*V)
        new_V = np.sum(np.multiply(policy, Q), axis=1)
        diff = np.abs(V - new_V)
        V = new_V

        if np.max(diff) < accuracy:
            converged = True
    return V


def get_optimal_policy(agent, gamma, accuracy):
    transitions = agent.tSAS
    rewards = agent.R
    Q = np.zeros((agent.size_environment, agent.size_actions))
    converged = False
    while not converged:
        max_Q = np.max(Q, axis=1)
        Q_new = rewards + gamma * np.dot(transitions, max_Q)
        diff = np.abs(Q - Q_new)
        Q = Q_new
        if diff.max() < accuracy:
            converged = True
    deterministic_policy = np.argmax(Q+1e-5*np.random.random(Q.shape), axis=1)  # random tie breaking
    policy = np.zeros((25, 5))
    policy[np.arange(25), deterministic_policy] = 1
    return policy


def get_agent_current_policy(agent):
    policy = np.zeros((25, 5))
    if type(agent).__name__ == 'EpsilonMB':
        policy += agent.epsilon/5
        best_actions = np.argmax(agent.Q+1e-5*np.random.random(agent.Q.shape), axis=1)
        policy[np.arange(25), best_actions] += 1-agent.epsilon
    else:
        policy[np.arange(25), np.argmax(agent.Q+1e-5*np.random.random(agent.Q.shape), axis=1)] = 1
    return policy
