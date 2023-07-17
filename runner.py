import os
import ray
import time
import numpy as np
import torch
from torch import Tensor
from replay_buffer import BigBuffer


@ray.remote(num_cpus=1, num_gpus=1)
class Learner(object):
    def __init__(self, args, batch_size, mini_batch_size):
        self.args = args
        self.total_steps = 0
        self.agent = args.agent_class(args=args, batch_size=batch_size, mini_batch_size=mini_batch_size, agent_type="Learner")
        self.cwd = args.save_cwd
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)
        self.learner_device = torch.device(args.learner_device)
        print("Learner is Activated")
        self.buffer = BigBuffer()

    def collect_buffer(self, worker_run_ref):
        self.buffer.reset()
        exp_r = 0
        exp_steps = 0
        worker_num = len(worker_run_ref)
        while len(worker_run_ref) > 0:
            worker_return_ref, worker_run_ref = ray.wait(worker_run_ref, num_returns=1, timeout=0.1)
            if len(worker_return_ref) > 0:
                reward, buffer_items, steps = ray.get(worker_return_ref)[0]
                exp_r += reward
                exp_steps += steps
                self.buffer.add_mini_buffer(buffer_items)
        return exp_r / worker_num, exp_steps
    
    def compute_and_get_gradients(self, total_steps):
        '''agent update network using training data'''
        torch.set_grad_enabled(True)
        object_c, object_a, actor_grad, critic_grad = self.agent.train(self.buffer, total_steps)
        torch.set_grad_enabled(False)
        return (object_c, object_a), actor_grad, critic_grad
    
    def save(self):
        '''save'''
        actor_ref = ray.put(self.agent.actor)
        critic_ref = ray.put(self.agent.critic)
        shared_net_ref = ray.put(self.agent.shared_net)
        return [actor_ref, critic_ref, shared_net_ref]
        
    def get_actor(self):
        return self.agent.actor

    def get_weights(self):
        actor_weights = self.agent.actor.get_weights()
        critic_weights = self.agent.critic.get_weights()
        return actor_weights, critic_weights
    
    def set_weights(self, actor_weights, critic_weights):
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)
    
    def set_gradients_and_update(self, actor_grad, critic_grad, total_steps):
        self.agent.ac_optimizer.zero_grad()
        self.agent.actor.set_gradients(actor_grad, self.learner_device)
        self.agent.critic.set_gradients(critic_grad, self.learner_device)
        self.agent.ac_optimizer.step()
        if self.args.use_lr_decay:
            self.agent.lr_decay(total_steps)


@ray.remote(num_cpus=1, num_gpus=0.01)
class Worker(object):
    def __init__(self, worker_id: int, args):
        self.worker_id = worker_id
        self.args = args
        self.env = args.env_class()
        self.agent = args.agent_class(args, batch_size=None, mini_batch_size=None, agent_type="Worker")
        self.sample_epi_num = args.sample_epi_num

    def run(self, actor_weights, critic_weights):
        worker_id = self.worker_id
        torch.set_grad_enabled(False)
        '''init agent'''
        # agent.load_pretrain_model(args.pretrain_model_cwd)
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)

        '''loop'''
        '''Worker send the training data to Learner'''
        exp_reward, buffer_items, steps = self.agent.explore_env(self.env, self.sample_epi_num)
        return exp_reward, buffer_items, steps


@ray.remote(num_cpus=1, num_gpus=0.01)
class EvaluatorProc(object):
    def __init__(self, args):
        self.env = args.env_class()  # the env for Evaluator, `eval_env = env` in default
        self.agent = args.agent_class(args, batch_size=None, mini_batch_size=None, agent_type='Evaluator')
        self.agent_id = 0
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = args.evaluate_times  # number of times that get episodic cumulative return
        self.break_step = args.max_train_steps
        self.num_layers = args.num_layers  # number of rnn hidden layers
        self.rnn_hidden_dim = args.rnn_hidden_dim  # 
        del args

        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.max_r = -np.inf
        print("| Evaluator:"
              "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'Time':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'objA':>7}{'etc.':>7}")
        
    def run(self, actor_weights, critic_weights, total_step, exp_r, logging_tuple):
        torch.set_grad_enabled(False)

        '''loop'''
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)
        ref_list = self.evaluate_and_save(total_step, exp_r, logging_tuple)
        '''Evaluator send the training signal to Learner'''
        if_train = self.total_step <= self.break_step
        '''Evaluator save the training log and draw the learning curve'''
        print(f'| TrainingTime: {time.time() - self.start_time:>7.0f}')
        return [if_train, ref_list]
    
    def evaluate_and_save(self, new_total_step: int, exp_r: float, logging_tuple: tuple):
        self.total_step = new_total_step
        rewards_step_ten = self.get_rewards_and_step()
        returns = rewards_step_ten[:, 0]  # episodic cumulative returns of an
        steps = rewards_step_ten[:, 1]  # episodic step number
        avg_r = returns.mean().item()
        # std_r = returns.std().item()
        std_r = 0
        avg_s = steps.mean().item()
        # std_s = steps.std().item()
        std_s = 0

        train_time = int(time.time() - self.start_time)

        '''record the training information'''
        self.recorder.append((self.total_step, avg_r, std_r, exp_r, *logging_tuple))  # update recorder

        '''print some information to Terminal'''
        prev_r = self.max_r
        self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        print(f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
              f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
              f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in logging_tuple)}")
            
        if_save = self.max_r >= prev_r
        
        if if_save:
            actor_ref = ray.put(self.agent.actor)
            critic_ref = ray.put(self.agent.critic)
            shared_net_ref = ray.put(self.agent.shared_net)
            recorder_ref = ray.put(self.recorder)
            return [actor_ref, critic_ref, shared_net_ref, recorder_ref]
        else:
            return []

    def get_recorder(self):
        return self.recorder

    def get_rewards_and_step(self) -> Tensor:
        rewards_steps_list = list()
        for _ in range(self.eval_times):
            rewards_steps = evaluate(
                env=self.env, 
                actor=self.agent.actor, 
                num_layers=self.num_layers, 
                rnn_hidden_dim=self.rnn_hidden_dim
            )
            rewards_steps_list.append(rewards_steps)
        rewards_steps_ten = torch.tensor(rewards_steps_list, dtype=torch.float32)
        return rewards_steps_ten  # rewards_steps_ten.shape[1] == 2


"""util"""
def evaluate(env, actor, num_layers, rnn_hidden_dim):  #
    episode_reward = 0
    device = next(actor.parameters()).device
    state = env.reset()
    actor_hidden_state = torch.zeros(size=(num_layers, 1, rnn_hidden_dim), dtype=torch.float32, device=device)
    for step in range(env.time_limits):
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        a, actor_hidden_state = actor.choose_action(state.unsqueeze(0).unsqueeze(0), actor_hidden_state, True)
        state, r, done = env.step(a.detach().cpu().numpy()[0])  # Take a step
        episode_reward += r

        if done:
            break

    return episode_reward, step


def draw_learning_curve(recorder: np.ndarray = None,
                        fig_title: str = 'learning_curve',
                        save_path: str = 'learning_curve.jpg'):
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()
    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(fig_title, y=2.3)
    plt.savefig(save_path)
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
    