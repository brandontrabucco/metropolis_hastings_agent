import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym import wrappers

MY_SEED = 126
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.set_random_seed(MY_SEED)

class OnPolicyMHAgent(object):
    def __init__(self, action_space, state_space, temperature=0.1, cooling_rate=.999):
        """Implements the Metropolis Hastings Sampler."""
        self.states  = tf.placeholder(tf.float32, name="states",  shape=[None, sum(state_space.shape)])
        self.rewards = tf.placeholder(tf.float32, name="rewards", shape=[None])
        self.expected_reward = tf.reduce_mean(self.rewards)
        self.old_reward = tf.get_variable("old_reward", shape=self.expected_reward.shape, 
            initializer=tf.zeros_initializer(), trainable=False)
        self._keep_reward = tf.assign(self.old_reward, self.expected_reward)
        self.logits = tf.layers.dense(self.states, action_space.n)
        self.samples = tf.squeeze(tf.multinomial(self.logits, 1), axis=1)
        self.current_variables = tf.trainable_variables()
        self.new_variables = [tf.get_variable(v.name[:-2] + "_new", shape=v.shape, 
            initializer=tf.zeros_initializer(), trainable=False) for v in self.current_variables]
        self.old_variables = [tf.get_variable(v.name[:-2] + "_old", shape=v.shape, 
            initializer=tf.zeros_initializer(), trainable=False) for v in self.current_variables]
        self.log_prior_ratio = (
            sum([tf.nn.l2_loss(v) for v in self.old_variables]) - 
            sum([tf.nn.l2_loss(v) for v in self.new_variables]))
        self.temperature = tf.get_variable("temperature", dtype=tf.float32, initializer=temperature)
        self._cooling_step = tf.assign(self.temperature, self.temperature * cooling_rate)
        self.probability_to_accept = tf.exp(0.0 * self.log_prior_ratio + ((
            self.expected_reward - self.old_reward) / self.temperature))
        self._propose = tf.group(*[tf.assign(v1, v2 + tf.truncated_normal(v2.shape, mean=.0, stddev=.1)) 
            for v1, v2 in zip(self.new_variables, self.current_variables )])
        self._accept = tf.group(*[tf.assign(v1, v2) 
            for v1, v2 in zip(self.current_variables, self.new_variables )])
        self._backup = tf.group(*[tf.assign(v1, v2) 
            for v1, v2 in zip(self.old_variables, self.current_variables )])
        self._reject = tf.group(*[tf.assign(v1, v2) 
            for v1, v2 in zip(self.current_variables, self.old_variables )])
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
    def reset(self):
        """Initializes the neural network weights and bases."""
        self.sess.run(self.init_op)
    def propose(self):
        """Proposes new policy parameters using the proposal distribution."""
        self.sess.run(self._propose)
        self.sess.run(self._backup)
        self.sess.run(self._accept)
    def act(self, states):
        """Args: states,  a  float32 tensor shape [None, state_size].
        Returns: actions, an int32   tensor shape [None]."""
        states = np.array(states)
        assert(len(states.shape) == 2 and states.shape[1] == self.states.shape[1])
        return self.sess.run(self.samples, feed_dict={self.states: states})
    def step(self, states, actions, rewards, next_states):
        """Args: state,      a float32 tensor shape [None, state_size].
                 action,     a int32   tensor shape [None].
                 reward,     a float32 tensor shape [None].
                 next_state, a float32 tensor shape [None, state_size].
        Returns: None."""
        states, actions, rewards, next_states = (np.array(states), 
            np.array(actions), np.array(rewards), np.array(next_states))
        assert(len(states.shape)  == 2 and states.shape[1]  == self.states.shape[1])
        assert(len(actions.shape) == 1 and states.shape[0]  == actions.shape[0])
        assert(len(rewards.shape) == 1 and actions.shape[0] == rewards.shape[0])
        assert(len(next_states.shape)  == 2 and next_states.shape[1]  == self.states.shape[1])
        p = self.sess.run(self.probability_to_accept, feed_dict={self.rewards: rewards})
        if random.random() < p:
            self.sess.run(self._keep_reward, feed_dict={self.rewards: rewards})
        else:
            self.sess.run(self._reject)
        self.sess.run(self._cooling_step)

class BufferOfSamples(object):
    def __init__(self, capacity, batch_size, use_future=True):
        """Implements a buffer of samples from the Environment."""
        self.capacity = capacity
        self.batch_size = batch_size
        self.use_future = use_future
        self.empty()
    def empty(self):
        """Clears the current buffer of samples."""
        self._buffer = []
        self._episode = []
    def __len__(self):
        """Returns: the total length of the buffer."""
        return len(self._buffer) + len(self._episode)
    def ratio(self):
        """Returns: the fraction of the buffer that is used."""
        return len(self) / self.capacity
    def is_full(self):
        """Returns: whether the bufefr is at capacity."""
        return len(self) >= self.capacity
    def sample(self):
        """Returns: a list of (state, action, reward) tuples"""
        return random.sample(self._buffer, self.batch_size)
    def shrink(self):
        """Shrinks the buffer by one sample if necessary."""
        if self.is_full():
            if len(self._buffer) > 0:
                self._buffer = self._buffer[1:]
            else:
                self._episode = self._episode[1:]
    def add(self, state, action, reward, next_state):
        """Args: state,      a float32 tensor shape [state_size].
                 action,     a int32   tensor shape [].
                 reward,     a float32 tensor shape [].
                 next_state, a float32 tensor shape [state_size].
        Returns: None."""
        self.shrink()
        if self.use_future:
            for content in self._episode:
                content[2] = content[2] + reward
        self._episode = self._episode + [[state, action, reward, next_state]]
    def episode(self):
        """Flags that an episode of samples has finished being collected."""
        self._buffer = self._buffer + self._episode
        self._episode = []

def plot(*means_stds_name, title="", xlabel="", ylabel=""):
    """Generate a colorful plot with the provided data."""
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for indices, means, stds, name in means_stds_name:
        ccolor = np.random.rand(3,)
        plt.fill_between(indices, means - stds, means + stds, color=np.hstack([ccolor, [0.2]]))
        plt.plot(indices,means, color=np.hstack([ccolor, [1.0]]), label=name)
    plt.legend(loc=4)
    plt.savefig(title + ".png")
    plt.close()

def main(args):
    """Run a simulation of the specified task in openAI gym."""
    env = gym.make(args.env_id)
    env.seed(0)
    agent = OnPolicyMHAgent(env.action_space, env.observation_space, 
        temperature=args.temperature, cooling_rate=args.cooling_rate)
    buffer = BufferOfSamples(args.buffer_size, args.batch_size)
    logged_trials = []
    for t in range(args.num_trials):
        logged_rewards = []
        agent.reset()
        for i in range(args.training_steps):
            agent.propose()
            while not buffer.is_full():
                state = env.reset()
                done = False
                while not done and not buffer.is_full():
                    action = agent.act([state])[0]
                    next_state, reward, done, _info = env.step(action)
                    buffer.add(state, action, reward, next_state)
                    state = next_state
                    print("\rBuffer fill ratio: {0:.2f} / {1:.2f}".format(buffer.ratio(), 1.0), end="\r")
                buffer.episode()
            states, actions, rewards, next_states = zip(*buffer.sample())
            agent.step(states, actions, rewards, next_states)
            buffer.empty()
            logged_rewards.append(np.mean(rewards))
            print("On training step {0} average utility was {1}.".format(i, logged_rewards[-1]))
        logged_trials.append(logged_rewards)
    env.close()
    trajectories = np.array(logged_trials)
    plot(
        (np.arange(args.training_steps), np.mean(trajectories, axis=0), 
            np.std(trajectories, axis=0), "On Policy Metropolis-Hastings"),
        title="Sampling On Policy On {0}".format(args.env_id), 
        xlabel="Iteration", 
        ylabel="Expected Future Reward")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='CartPole-v0')
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cooling_rate", type=float, default=0.9)
    main(parser.parse_args())
