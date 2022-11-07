import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size,T, sample_func):
        self.env_params = env_params
        self.T = T
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        # self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
        #                 'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
        #                 'g': np.empty([self.size, self.T, self.env_params['goal']]),
        #                 'actions': np.empty([self.size, self.T, self.env_params['action']]),
        #                 }
         # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in env_params.items()}
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):

        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T
    




        ##---------------
        # mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        # batch_size = mb_obs.shape[0]
        # with self.lock:
        #     idxs = self._get_storage_idx(inc=batch_size)
        #     # store the informations
        #     self.buffers['obs'][idxs] = mb_obs
        #     self.buffers['ag'][idxs] = mb_ag
        #     self.buffers['g'][idxs] = mb_g
        #     self.buffers['actions'][idxs] = mb_actions
        #     self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    # def sample(self, batch_size):
    #     temp_buffers = {}
    #     with self.lock:
    #         for key in self.buffers.keys():
    #             temp_buffers[key] = self.buffers[key][:self.current_size]
    #     temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
    #     temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
    #     # sample transitions
    #     transitions = self.sample_func(temp_buffers, batch_size)
    #     return transitions
    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['obs_next'] = buffers['obs'][:, 1:, :]
        buffers['ag_next'] = buffers['ag'][:, 1:, :]
        buffers['obs_image_next'] = buffers['obs_image'][:,1:,:]
        transitions = self.sample_func(buffers, batch_size)

        for key in (['r', 'obs_next', 'ag_next','obs_image_next'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0