import numpy as np
import torch


class MiniBuffer:
    def __init__(self, episode_limit, sample_epi_num, agent_num,worker_device):
        self.episode_limit = episode_limit
        self.batch_size = sample_epi_num
        self.max_episode_len = 0
        self.buffer = None
        self.agent_num = agent_num
        self.device = torch.device(worker_device)

    def reset_buffer(self, obs_dim, action_dim):
        def zero_ten(size):
            tensor = torch.zeros(size=size, dtype=torch.float32, device=self.device)
            return tensor
        self.buffer = dict()
        self.buffer['state'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit, obs_dim))
        self.buffer['v'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit + 1, 1))
        self.buffer['a'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit, action_dim))
        self.buffer['a_logprob'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit, action_dim))
        self.buffer['r'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit, 1))
        self.buffer['active'] = zero_ten(size=(self.agent_num,self.batch_size, self.episode_limit, 1))

        self.max_episode_len = 0

    def store_transition(self, num_episode, episode_step, state, v, a, a_logprob, r, done,agent_id):
        self.buffer['state'][agent_id][num_episode][episode_step] = state
        self.buffer['v'][agent_id][num_episode][episode_step] = v
        self.buffer['a'][agent_id][num_episode][episode_step] = a
        self.buffer['a_logprob'][agent_id][num_episode][episode_step] = a_logprob
        self.buffer['r'][agent_id][num_episode][episode_step] = r
        self.buffer['active'][agent_id][num_episode][episode_step] = done

    def store_last_value(self, num_episode, episode_step, agent_id, v):
        self.buffer['v'][agent_id][num_episode][episode_step] = v
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step


class BigBuffer:
    def __init__(self):
        self.buffer = None
        
    def add_mini_buffer(self, mini_buffer):  # TODO: Match mini_buffer with worker_id
        if self.buffer is None:
            self.buffer = mini_buffer
        else:
            self.concat_mini_buffer(mini_buffer)

    def get_training_data(self, device):
        buffer = self.buffer.buffer
        max_episode_len = self.buffer.max_episode_len
        for key in buffer:
            if key == 'v':
                buffer[key] = buffer[key][:,:, :max_episode_len + 1].to(device)
            else:
                buffer[key] = buffer[key][:,:, :max_episode_len].to(device)
        return buffer, max_episode_len

    def reset(self):
        self.buffer = None
        
    def concat_mini_buffer(self, mini_buffer):
        for key in self.buffer.buffer:
            self.buffer.buffer[key] = torch.cat([self.buffer.buffer[key], mini_buffer.buffer[key]], dim=1)
        
        self.buffer.max_episode_len = max(self.buffer.max_episode_len, mini_buffer.max_episode_len)