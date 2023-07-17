import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as f
from torch.utils.data.sampler import *
from torch.nn.utils import spectral_norm
from replay_buffer import MiniBuffer
from normalization import Normalization


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


def preproc_layer(input_size, output_size, is_sn=False):
    layer = nn.Linear(input_size, output_size)
    orthogonal_init(layer)
    return spectral_norm(layer) if is_sn else layer


class MlpLstmExtractor(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dim, mlp_output_dim, rnn_hidden_dim, num_rnn_layers):
        super().__init__()
        self.fc1 = nn.Sequential(
            preproc_layer(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            preproc_layer(mlp_hidden_dim, mlp_output_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=input_dim + mlp_hidden_dim, 
            hidden_size=rnn_hidden_dim, 
            num_layers=num_rnn_layers,
            batch_first=True
        )
    
    def forward(self, state, hidden_state):
        s = self.fc1(state)
        s = torch.concatenate((s, state), dim=-1)
        s, hidden_state = self.gru(s, hidden_state)
        s = torch.concatenate((s, state), dim=-1)
        return s, hidden_state
    

# Discrete Actor
class SharedActor(nn.Module):
    def __init__(self, shared_net, input_size, action_dim, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.Mean = preproc_layer(input_size, action_dim) if is_sn else nn.Linear(input_size, action_dim)

    def forward(self, state, hidden_state):
        # When choose action:
        #             state          : tensor of the shape (Obs_Size)
        # For MLP:
        #             state          : tensor of the shape (Obs_Size) => (1(Batch), 1(Length), Obs_Size)
        #             output(ResNet) : tensor of the shape (1(Batch), 1(Length), MLP_Hidden_Size + Obs_Size)
        # For GRU: 
        #             unbatched input: tensor of the shape (1(Batch), 1(Length), MLP_Hidden_Size + Obs_Size)
        #             hidden_state   : tensor of the shape (1(Bidirectional or Not) * Num_layers, RNN_Hidden_Size)
        #             output(ResNet) : tensor of the shape (1(Batch), 1(Length), RNN_Hidden_Size + Obs_Size)   
        # For policy head:
        #             input          : tensor of the shape (1(Batch), 1(Length), RNN_Hidden_Size + Obs_Size)
        #             output         : tensor of the shape (1(Batch), 1(Length), Action_Dim)        
        #################################################################################################################################################################
        # When get logprob & entropy: 
        #             state          : tensor of the shape (Batch, Steps, Obs_Size)
        # For MLP:
        #             state          : tensor of the shape (Batch, Steps, Obs_Size)
        #             output         : tensor of the shape (Batch, Steps, MLP_Hidden_Size + Obs_Size)
        # For GRU: 
        #             input          : tensor of the shape (Batch, Steps(Length), MLP_Hidden_Size + Obs_Size)
        #             hidden_state   : tensor of the shape (1(Bidirectional or Not) * Num_layers, Batch, RNN_Hidden_Size)
        #             output(ResNet) : tensor of the shape (Batch, Steps(Length), RNN_Hidden_Size + Obs_Size)
        # For policy head:
        #             input          : tensor of the shape (Batch, Steps(Length), RNN_Hidden_Size + Obs_Size)
        #             output         : tensor of the shape (Batch, Steps(Length), Action_Dim)
        s, hidden_state = self.shared_net(state, hidden_state)
        mean = self.Mean(s)
        mean = torch.softmax(mean, dim=-1)  # [-1,1]->[-max_action,max_action]
        return mean, hidden_state
    
    def choose_action(self, state, hidden_state, deterministic=True):
        mean, hidden_state = self.forward(state, hidden_state)
        if deterministic:
            action = mean.argmax(dim=-1, keepdim=False)
            return action.flatten(), hidden_state
        else:
            dist = Categorical(mean)  # Get the Categorical distribution
            a = dist.sample()  # Sample the action according to the probability distribution
            a_logprob = dist.log_prob(a)  # The log probability density of the action
            return a.flatten(), a_logprob.flatten(), hidden_state

    def get_logprob_and_entropy(self, state, hidden_state, action):
        mean, _ = self.forward(state, hidden_state)
        dist = Categorical(mean)  # Get the Categorical distribution
        a_logprob = dist.log_prob(action.squeeze(-1))
        entropy = dist.entropy()
        return a_logprob, entropy  # tensor of the shape (Batch, Steps(Length), Action_Dim)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.tensor(g).to(device)


class SharedCritic(nn.Module):
    def __init__(self, shared_net, input_size, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.Value = preproc_layer(input_size, 1, is_sn=is_sn)

    def forward(self, state, hidden_state):
        # When obtain single value:
        #             state          : tensor of the shape (Obs_Size)
        # For MLP:
        #             state          : tensor of the shape (Obs_Size) => (1, Obs_Size)
        #             output(ResNet) : tensor of the shape (1, MLP_Hidden_Size + Obs_Size)
        # For GRU: 
        #             unbatched input: tensor of the shape (1(Length), MLP_Hidden_Size + Obs_Size)
        #             hidden_state   : tensor of the shape (1(Bidirectional or Not) * Num_layers, RNN_Hidden_Size)
        #             output(ResNet) : tensor of the shape (1(Length), RNN_Hidden_Size + Obs_Size)   
        # For value head:
        #             input          : tensor of the shape (1(Length), RNN_Hidden_Size + Obs_Size)
        #             output         : tensor of the shape (1(Length), 1)        
        #################################################################################################################################################################
        # When obtain batch value: 
        #             state          : tensor of the shape (Batch, Steps, Obs_Size)
        # For MLP:
        #             state          : tensor of the shape (Batch, Steps, Obs_Size)
        #             output         : tensor of the shape (Batch, Steps, MLP_Hidden_Size + Obs_Size)
        # For GRU: 
        #             input          : tensor of the shape (Batch, Steps(Length), MLP_Hidden_Size + Obs_Size)
        #             hidden_state   : tensor of the shape (1(Bidirectional or Not) * Num_layers, Batch, RNN_Hidden_Size)
        #             output(ResNet) : tensor of the shape (Batch, Steps(Length), RNN_Hidden_Size + Obs_Size)
        # For value head:
        #             input          : tensor of the shape (Batch, Steps(Length), RNN_Hidden_Size + Obs_Size)
        #             output         : tensor of the shape (Batch, Steps(Length), 1)   
        s, hidden_state = self.shared_net(state, hidden_state)
        value = self.Value(s)
        return value, hidden_state

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.tensor(g).to(device)


class PPO_discrete:
    def __init__(self, args, batch_size, mini_batch_size, agent_type):
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.num_layers = args.num_layers
        self.state_dim = args.state_dim
        self.action_dim = 1
        self.discrete_action_dim = args.discrete_action_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        if "Learner" in agent_type:
            self.device = torch.device(args.learner_device)
        elif "Worker" in agent_type:
            self.device = torch.device(args.worker_device)
        else:
            self.device = torch.device(args.evaluator_device)

        if args.use_reward_norm:
            self.reward_norm = Normalization(shape=1)

        self.shared_net = MlpLstmExtractor(
            input_dim=args.state_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_output_dim=args.mlp_output_dim,
            rnn_hidden_dim=args.rnn_hidden_dim,
            num_rnn_layers=args.num_layers
        ).to(self.device)

        self.actor = SharedActor(
            shared_net=self.shared_net,
            input_size=self.rnn_hidden_dim + self.state_dim, 
            action_dim=self.discrete_action_dim, 
            is_sn=True
        ).to(self.device)
        
        self.critic = SharedCritic(
            shared_net=self.shared_net,
            input_size=self.rnn_hidden_dim + self.state_dim, 
            is_sn=True
        ).to(self.device)

        self.ac_parameters = list(self.shared_net.parameters()) + list(self.actor.Mean.parameters())+ list(self.critic.Value.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        
        self.minibuffer = None
        self.args = args

    def train(self, replay_buffer, total_steps):
        self.actor = self.actor.to(self.device)        
        batch, max_episode_len = replay_buffer.get_training_data(self.device)  # Transform the data into tensor

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,1)
            deltas = batch['r'] + self.gamma * batch['v'][:, 1:] - batch['v'][:, :-1]
            deltas = deltas * batch['active']
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,1)
            v_target = adv + batch['v'][:, :-1]  # v_target.shape(batch_size,max_episode_len,1)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                mean = adv.mean()
                std = adv.std()
                adv = (adv - mean) / (std + 1e-5) * batch['active']
        object_critics = 0.0
        object_actors = 0.0
        update_time = 0

        self.ac_optimizer.zero_grad()
        
        batch, _ = replay_buffer.get_training_data(self.device)  # Transform the data into tensor
        for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
            actor_hidden_state = torch.zeros(
                size=(self.num_layers, self.mini_batch_size, self.rnn_hidden_dim),
                dtype=torch.float32,
                device=self.device
            )
            critic_hidden_state = torch.zeros(
                size=(self.num_layers, self.mini_batch_size, self.rnn_hidden_dim),
                dtype=torch.float32,
                device=self.device
            )
            a_logprob_now, dist_entropy = self.actor.get_logprob_and_entropy(batch['state'][index], actor_hidden_state, batch['a'][index])
            values_now, _ = self.critic(batch['state'][index], critic_hidden_state)
            ratios = torch.exp(a_logprob_now.unsqueeze(-1) - batch['a_logprob'][index].detach())  # Attention! Attention! 'a_log_prob_n' should be detached.
            # dist_entropy.shape=(mini_batch_size, max_episode_len, 1)
            # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, 1)
            # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, 1)
            # ratios.shape=(mini_batch_size, max_episode_len, 1)
            surr1 = ratios * adv[index]
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy.unsqueeze(-1)
            actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

            if self.use_value_clip:
                values_old = batch["v"][index, :-1].detach()
                values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                values_error_original = values_now - v_target[index]
                critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
            else:
                critic_loss = (values_now - v_target[index]) ** 2
            critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
            
            ac_loss = actor_loss + critic_loss
            ac_loss.backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
            
            object_critics += critic_loss.item()
            object_actors += actor_loss.item()
            update_time += 1
                
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        actor_gradients = self.actor.get_gradients()
        critic_gradients = self.critic.get_gradients()
        
        return object_critics / update_time, object_actors / update_time, actor_gradients, critic_gradients
    
    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now
    
    def explore_env(self, env, num_episode):
        exp_reward = 0.0
        sample_steps = 0
        self.minibuffer = MiniBuffer(episode_limit=self.args.episode_limit, sample_epi_num=num_episode, worker_device=self.device)
        self.minibuffer.reset_buffer(obs_dim=self.state_dim, action_dim=self.action_dim)
        for k in range(num_episode):
            episode_reward, episode_steps = self.run_episode(env, num_episode=k)
            exp_reward += episode_reward
            sample_steps += episode_steps
        return exp_reward / num_episode, self.minibuffer, sample_steps

    def run_episode(self, env, num_episode=0):  #
        episode_reward = 0
        state = env.reset()
        actor_hidden_state = torch.zeros(size=(self.num_layers, 1, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        critic_hidden_state = torch.zeros(size=(self.num_layers, 1, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        for episode_step in range(self.args.episode_limit):
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            a, a_logprob, actor_hidden_state = self.actor.choose_action(state.unsqueeze(0).unsqueeze(0), actor_hidden_state, False)
            v, critic_hidden_state = self.critic(state.unsqueeze(0).unsqueeze(0), critic_hidden_state)  # Get the state values (V(s)) of N agents
            v = v.flatten()
            next_state, r, done = env.step(a.detach().cpu().numpy()[0])  # Take a step

            episode_reward += r
            r = self.reward_norm(r)
            
            # Store the transition
            r = torch.as_tensor(r, dtype=torch.float32).to(self.device)
            dw = torch.as_tensor(1 - done, dtype=torch.float32).to(self.device)
            self.minibuffer.store_transition(num_episode, episode_step, state, v, a, a_logprob, r, dw)
            state = next_state
            if done:
                break

        # An episode is over, store obs_n, s and avail_a_n in the last step
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        v, _ = self.critic(state.unsqueeze(0).unsqueeze(0), critic_hidden_state)
        v = v.flatten()
        self.minibuffer.store_last_value(num_episode, episode_step + 1, v)
        return episode_reward, episode_step + 1

    def save_model(self, cwd):
        torch.save(self.actor.state_dict(), cwd + 'actor.pth')
        torch.save(self.critic.state_dict(), cwd + 'critic.pth')
