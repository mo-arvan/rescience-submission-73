"""Contains all the policy related functions."""
import numpy as np


def value_iteration(environment, gamma=0.95, accuracy=1e-3):
    """
    Return the value iteration after convergence and the optimal policy.

    Parameters
    ----------
    environment : Lopes_environment
    gamma : float (between 0 and 1)
    accuracy : float

    Returns
    -------
    V : numpy.ndarray of shape (size_state)
    policy : numpy.ndarray of shape (size_state, size_action)
    """
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


def policy_evaluation(environment, policy, gamma=0.95, accuracy=1e-3):
    """
    Evaluate a :policy: on :environment: and return the values after value iteration.

    Parameters
    ----------
    environment : Lopes_environment
    policy : numpy.ndarray of shape (size_state, size_action)
    gamma : float (between 0 and 1)
    accuracy : float (>0)

    Returns
    -------
    V : numpy.ndarray of shape (size_state)
    """
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


def get_optimal_policy(agent, gamma=0.95, accuracy=1e-3):
    """
    Return the theoretical optimal policy on the environment sampled by :agent:.

    Parameters
    ----------
    agent : one of the classes of the 'agents.py' file
    gamma : float (between 0 and 1)
    accuracy : float (>0)

    Returns
    -------
    policy : numpy.ndarray of shape (size_state, size_action)

    """
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
    deterministic_policy = np.argmax(Q+1e-5*np.random.random(Q.shape), axis=1)  # tie breaking
    policy = np.zeros((25, 5))
    policy[np.arange(25), deterministic_policy] = 1
    return policy


def get_agent_current_policy(agent):
    """Return the policy used by :agent: in the decision-making process."""
    policy = np.zeros((25, 5))
    if type(agent).__name__ == 'EpsilonMB':
        policy += agent.epsilon/5
        best_actions = np.argmax(agent.Q+1e-5*np.random.random(agent.Q.shape), axis=1)
        policy[np.arange(25), best_actions] += 1-agent.epsilon
    else:
        policy[np.arange(25), np.argmax(agent.Q+1e-5*np.random.random(agent.Q.shape), axis=1)] = 1
    return policy
