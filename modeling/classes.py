import numpy as np
import random


class EnvironmentRewardsPairsTask:
    """Environment for the Rewards Pairs task.

    Attributes:
        n_stimuli: total number of stimuli
        rewards: list of rewards for each stimulus
        n_actions: number of actions available
        frequency_ratio: ratio of the frequency of frequently vs rarely chosen stimuli
        seed: seed value for random number generator
    """

    def __init__(
                self,
                frequency_ratio: int=3,
                seed: int=None):
        """Initialize the environment."""        

        # Initialize properties
        self.frequency_ratio = frequency_ratio
        self.seed = seed

        # Stuff, where to put?
        self.stimuli = [f"S{i}" for i in range(8)]
        self.rewards = [1, 3, 3, 5, 5, 7, 7, 9]
        self.stimuli_rewards = dict(zip(self.stimuli, self.rewards))
        self.choice_pairs = [(0,1),(0,2),(1,4),(2,3),(4,5),(3,6),(5,7),(6,7)]

        # Create the trial pairs
        self._create_trial_pairs()
        self._reset_iterator()

    def _create_trial_pairs(self):
        ratio = self.frequency_ratio
        self.pairs_repeats = [1, ratio, ratio, 1, 1, ratio, ratio, 1]
        self.repeated_pairs = [pair for pair, repeat in zip(self.choice_pairs, self.pairs_repeats) for _ in range(repeat)]

    def reset(self):
        self._reset_iterator()
    
    def _reset_iterator(self):
        shuffled_pairs = self.repeated_pairs.copy()
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(shuffled_pairs)
        self._pair_iterator = iter(shuffled_pairs)

    def present_stimuli(self) -> tuple:
        """Select and present a pair of stimuli.

        Returns:
            presented_pair: The pair of stimuli presented.
        """
        # Randomly select a pair of stimuli
        try: 
            self._current_pair = next(self._pair_iterator)
        except StopIteration:
            self._reset_iterator()
            self._current_pair = next(self._pair_iterator)
        return self._current_pair

    def get_reward(self, choice: int) -> int:
        """Process the agent's choice and return the reward.

        Args:
            choice: the chosen stimuli.

        Returns:
            reward: The reward associated with the chosen stimuli.
        """
        if self._current_pair is None:
            raise ValueError('Stimuli must be presented before making a choice.')

        # Get the reward value for the chosen stimuli
        reward = self.stimuli_rewards[f"S{choice}"]

        return reward
    

class AgentQ:
    """A vanilla Q-learning agent. Updates only the value of the chosen stimulus.
    """

    def __init__(
            self,
            alpha: float=0.1,
            beta: float=0.1,
            n_stimuli: int=8,
            n_actions: int=2
    ):
        # self._prev_choice = None this is to add some perseveration, we don't need it yet.
        self._alpha = alpha
        self._beta = beta
        self._n_stimuli = n_stimuli
        self._n_actions = n_actions
        self._q_init = 5 # corresponds to the median reward level

        self.new_sess()

    def new_sess(self):
        self._q = self._q_init * np.ones((self._n_stimuli,))

    def get_choice_probs(self, pair) -> np.ndarray:
        """Compute the choice probabilites as softmax over q of presented stimuli."""
        scaled_action_val = self._beta * self._q[list(pair)]
        choice_probs = np.exp(scaled_action_val) / np.sum(np.exp(scaled_action_val))
        return choice_probs
    
    def get_choice(self, pair) -> int:
        """Choose a stimulus from the presented pair based on q values."""
        choice_probs = self.get_choice_probs(pair)
        return np.random.choice(pair, p=choice_probs)
    
    def update(self,
                 choice: int,
                 reward: int):
        """Update the q value for the chosen stimulus.
        Args:
            choice: The chosen stimulus.
            reward: The reward associated with the chosen stimulus.
        """
        self._q[choice] += self._alpha * (reward - self._q[choice])

    @property
    def q(self):
        # This establishes q as an externally visible attribute of the agent.
        # For agent = AgentQ(...), you can view the q values with agent.q; however,
        # you will not be able to modify them directly because you will be viewing
        # a copy.
        return self._q.copy()
    
class AgentCK(AgentQ):
    def __init__(
            self,
            alpha_q: float=0.1,
            alpha_h: float=0.1,
            beta_q: float=0.1,
            beta_h: float=0.1,
            n_stimuli: int=8,
            n_actions: int=2
    ):
        # self._prev_choice = None this is to add some perseveration, we don't need it yet.
        self._alpha_q = alpha_q
        self._beta_q = beta_q
        self._alpha_h = alpha_h
        self._beta_h = beta_h
        self._n_stimuli = n_stimuli
        self._n_actions = n_actions
        self._q_init = 5 # corresponds to the median reward level

        self.new_sess()

    def new_sess(self):
        self._q = self._q_init * np.ones((self._n_stimuli,))
        self._h = np.zeros((self._n_stimuli,))

    def get_choice_probs(self, pair) -> np.ndarray:
        """Compute the choice probabilites as softmax over q of presented stimuli."""
        scaled_q_val = self._beta_q * self._q[list(pair)]
        scaled_h_val = self._beta_h * self._h[list(pair)]
        choice_probs = np.exp(scaled_q_val+scaled_h_val) / np.sum(np.exp(scaled_q_val+scaled_h_val))
        return choice_probs
    
    def update(self,
                 pair: tuple,
                 choice: int,
                 reward: int):
        """Update the q value for the chosen stimulus.
        Args:
            choice: The chosen stimulus.
            reward: The reward associated with the chosen stimulus.
        """
        self._q[choice] += self._alpha_q * (reward - self._q[choice])
        self._h[choice] += self._alpha_h * (1 - self._h[choice])

        not_chosen = (set(pair) - {choice}).pop()
        self._h[not_chosen] += self._alpha_h * (0 - self._h[not_chosen])

    @property
    def h(self):
        return self._h.copy()