import numpy as np


class BasicAgent:

    def __init__(self, environment, gamma=0.95):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.gamma = gamma

        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment, self.size_actions, self.size_environment)
        self.R = np.zeros(self.shape_SA)
        self.Rsum = np.zeros(self.shape_SA)
        self.R_VI = np.zeros(self.shape_SA)

        self.nSA = np.zeros(self.shape_SA, dtype=np.int32)
        self.nSAS = np.zeros(self.shape_SAS, dtype=np.int32)

        self.tSAS = np.ones(self.shape_SAS)/self.size_environment
        self.Q = np.ones(self.shape_SA)/(1-self.gamma)
        self.step_counter = 0

    def choose_action(self):
        self.step_counter += 1
        q_values = self.Q[self.environment.current_location]
        return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    def learn(self, old_state, reward, new_state, action):

        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward
        self.R[old_state][action] = self.Rsum[old_state][action] / self.nSA[old_state][action]

        self.compute_reward_VI(old_state, reward, action)
        self.compute_transitions(old_state, new_state, action)
        self.compute_learning_progress(old_state, new_state, action)

        self.value_iteration()

    def compute_transitions(self, old_state, new_state, action):
        self.tSAS[old_state][action] = self.nSAS[old_state][action]/self.nSA[old_state][action]

    def compute_reward_VI(self, old_state, reward, action):
        self.R_VI[old_state][action] = self.R[old_state][action]

    def value_iteration(self):
        # visited = np.where(self.nSA >= 0, 1, 0)  # computes Q-values for all (state,action)
        visited = np.where(self.nSA >= 1, 1, 0)  # computes Q-values only for visited (state,action)
        threshold = 1e-3
        converged = False
        while not converged:
            max_Q = np.max(self.Q, axis=1)
            new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

            diff = np.abs(self.Q[visited > 0] - new_Q[visited > 0])
            self.Q[visited > 0] = new_Q[visited > 0]
            if np.max(diff) < threshold:
                converged = True

    def compute_learning_progress(self, old_state, new_state, action):
        pass


class EpsilonMB(BasicAgent):

    def __init__(self, environment, gamma, epsilon):
        self.epsilon = epsilon
        super().__init__(environment, gamma)

    def choose_action(self):
        self.step_counter += 1
        if np.random.random() > (1 - self.epsilon):
            action = np.random.choice(self.environment.actions)
        else:
            q_values = self.Q[self.environment.current_location]
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return action


class Rmax(BasicAgent):

    def __init__(self, environment, gamma, Rmax, m, m_u, condition='informative'):
        super().__init__(environment, gamma)
        self.Rmax = Rmax
        self.m = m
        self.m_u = m_u
        self.R_VI = np.ones(self.shape_SA)*self.Rmax
        if condition == 'informative':
            self.max_visits = np.ones(self.shape_SA)*self.m
            for state in self.environment.uncertain_states:
                self.max_visits[state] = np.ones(self.size_actions)*self.m_u
        elif condition == 'wrong_prior':
            self.max_visits = np.random.randint(self.m, self.m_u, (self.shape_SA))
        elif condition == 'uninformative':
            self.max_visits = np.ones(self.shape_SA)*self.m
        else:
            raise ValueError("The condition " + str(condition) + " does not exist."
                             " The conditions are: informative, wrong_prior or uninformative.")

    def compute_reward_VI(self, old_state, reward, action):
        if self.nSA[old_state][action] >= self.max_visits[old_state][action]:
            self.R_VI[old_state][action] = self.R[old_state][action]
        else:
            self.R_VI[old_state][action] = self.Rmax


class BEB(BasicAgent):

    def __init__(self, environment, gamma, beta, coeff_prior, condition='informative'):
        super().__init__(environment, gamma)

        self.beta = beta
        self.coeff_prior = coeff_prior

        if condition == 'informative':
            self.prior = self.environment.transitions * self.coeff_prior + 1e-5
        elif condition == 'uninformative':
            self.prior = np.ones(self.shape_SAS) * self.coeff_prior
        elif condition == 'wrong_prior':
            max_prior = np.max(self.environment.transitions * self.coeff_prior + 1e-5)
            self.prior = np.random.uniform(1e-5, max_prior, (self.shape_SAS))
        else:
            raise ValueError("The condition "+str(condition)+" does not exist."
                             " The conditions are: informative, wrong_prior or uninformative")
        self.prior_0 = self.prior.sum(axis=2)
        self.bonus = np.ones(self.shape_SA) * self.beta / (1 + self.prior_0)
        self.Q = np.ones(self.shape_SA) * (1 + self.beta) / (1 - self.gamma)
        self.R_VI = self.R + self.bonus

    def compute_reward_VI(self, old_state, reward, action):
        self.prior_0[old_state][action] += 1
        self.bonus[old_state][action] = self.beta / (1 + self.prior_0[old_state][action])
        self.R_VI[old_state][action] = self.R[old_state][action] + self.bonus[old_state][action]

    def compute_transitions(self, old_state, new_state, action):
        self.prior[old_state][action][new_state] += 1
        self.tSAS[old_state][action] = np.random.dirichlet(self.prior[old_state][action])


class LearningProgress(BasicAgent):

    def __init__(self, environment, gamma, step_update, alpha, prior_LP):
        super().__init__(environment, gamma)
        self.step_update = step_update
        self.alpha = alpha
        self.prior_LP = prior_LP
        self.last_k = np.zeros((self.size_environment, self.size_actions, self.step_update),
                               dtype=np.int32)
        self.LP = np.ones(self.shape_SA) * 10

    def compute_learning_progress(self, old_state, new_state, action):
        self.last_k[old_state][action][self.nSA[old_state][action] % self.step_update] = new_state
        if self.nSA[old_state][action] > self.step_update:
            old_array = np.copy(self.nSAS[old_state][action])
            for last_seen_state in self.last_k[old_state][action]:
                old_array[last_seen_state] -= 1
            arrival_states = self.nSAS[old_state][action][self.nSAS[old_state][action] > 0]
            new_CV, new_variance = self.cross_validation(arrival_states)
            old_CV, old_variance = self.cross_validation(old_array[old_array > 0])
            learning_progress = old_CV-new_CV+self.alpha*np.sqrt(new_variance)
            self.LP[old_state][action] = max(learning_progress, 0.001)

    def cross_validation(self, array_of_state):
        sum_count = np.sum(array_of_state)
        sum_prior = sum_count + self.size_environment * self.prior_LP
        estimated_transition_probas = (array_of_state - 1 + self.prior_LP) / (sum_prior-1)
        log_values = np.log(estimated_transition_probas)
        cross_validation = -np.dot(log_values, array_of_state) / sum_count
        variance_cv = np.dot(array_of_state, (log_values+cross_validation) ** 2) / sum_count
        return cross_validation, variance_cv


class EBLP(LearningProgress):

    def __init__(self, environment, gamma, beta, step_update, alpha, prior_LP):

        super().__init__(environment, gamma, step_update, alpha, prior_LP)
        self.beta = beta
        self.bonus = np.ones(self.shape_SA) * self.beta / (1 + 1 / np.sqrt(self.LP))
        self.Q = np.ones(self.shape_SA) * (1 + self.beta) / (1 - self.gamma)
        self.R_VI = self.R + self.bonus

    def compute_reward_VI(self, old_state, reward, action):
        self.bonus[old_state][action] = self.beta / (1 + 1 / np.sqrt(self.LP[old_state][action]))
        self.R_VI[old_state][action] = self.R[old_state][action] + self.bonus[old_state][action]


class RmaxLP(LearningProgress):

    def __init__(self, environment, gamma, step_update, alpha, prior_LP, Rmax, m):

        super().__init__(environment, gamma, step_update, alpha, prior_LP)
        self.Rmax = Rmax
        self.m = m
        self.R_VI = np.ones(self.shape_SA) * self.Rmax

    def compute_reward_VI(self, old_state, reward, action):
        if self.LP[old_state][action] < self.m:
            self.R_VI[old_state][action] = self.R[old_state][action]
        else:
            self.R_VI[old_state][action] = self.Rmax
