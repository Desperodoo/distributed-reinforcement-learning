import os
import ray
import time
import warnings
# import wandb
import torch
import argparse
import numpy as np

from case3d_env import chase3D
from ppo_continuous import PPO_continuous
from ppo_discrete import PPO_discrete
from runner import Learner, Worker, EvaluatorProc, draw_learning_curve
from arguments import ppo_args

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# server 0
# ip: 172.18.166.252
# num_cpus: 80
# num_gpus: 5 * RTX 8000
# server 1
# ip: 172.18.192.190
# num_cpus: 112
# num_gpus: 3 * RTX 3090
# server 2
# ip: 172.18.196.189
# num_cpus: 72
# num_gpus: 4 * RTX 2080
# server 3
# ip: 172.18.196.180
# num_cpus: 32
# num_gpus: 4 * RTX 2080


def train_agent_multiprocessing(args):
    num_nodes = 2
    num_workers = [70, 30]
    # num_workers[0] = int(80 / 9)
    # num_workers[1] = int(112 / 9)
    # num_workers[0] = 70
    # num_workers[1] = 30
    
    mini_batch_size = [num_workers[node] * args.sample_epi_num for node in range(num_nodes)]
    batch_size = [num_workers[node] * args.sample_epi_num for node in range(num_nodes)]
    # implement environment information
    env = args.env_class()
    _ = env.reset()
    args.episode_limit = env.time_limits
    args.state_dim = env.state_size
    args.action_dim = 1
    args.discrete_action_dim = env.action_size
    cwd = args.save_cwd
    del env
    # build learners and workers
    learners = list()
    workers = [[] for _ in range(num_nodes)]
    worker_id = 0
    for node in range(num_nodes): 
        learners.append(Learner.options(resources={f"node_{node}": 0.001}).remote(args, batch_size[node], mini_batch_size[node]))
        for _ in range(num_workers[node]):
            workers[node].append(Worker.options(resources={f"node_{node}": 0.001}).remote(worker_id, args))
            worker_id += 1
    # build evaluators
    evaluator = EvaluatorProc.options(resources={f"node_{0}": 0.001}).remote(args)
    # initialize training
    if_Train = True
    total_steps = 0
    exp_r = 0
    eval_run_ref = None
    # initialize learners (get original weights from head node)
    actor_weights, critic_weights = ray.get(learners[0].get_weights.remote())
    for node in range(num_nodes):
        learners[node].set_weights.remote(actor_weights, critic_weights)
    # loop
    while if_Train:
        front_time = time.time()
        # send actor_weights to workers and workers sample     
        worker_run_ref = list()
        exp_r = 0
        exp_steps = 0
        for node in range(num_nodes):
            worker_run_ref.append([worker.run.remote(actor_weights, critic_weights) for worker in workers[node]])
        # learners receive training data from corresponding workers
        learner_run_ref = [learners[node].collect_buffer.remote(worker_run_ref[node]) for node in range(num_nodes)]
        while len(learner_run_ref) > 0:
            learner_ret_ref, learner_run_ref = ray.wait(learner_run_ref, num_returns=1, timeout=0.1)
            if len(learner_ret_ref) > 0:
                r, steps = ray.get(learner_ret_ref)[0]
                exp_r += r
                exp_steps += steps
        exp_r /= num_nodes
        total_steps += exp_steps
        # training for k_epochs
        # learners compute and send gradients to main func
        for _ in range(args.K_epochs):
            actor_gradients = []
            critic_gradients = []
            obj_c = 0
            obj_a = 0
            learner_run_ref = [learner.compute_and_get_gradients.remote(total_steps) for learner in learners]
            while len(learner_run_ref) > 0:
                learner_ret_ref, learner_run_ref = ray.wait(learner_run_ref, num_returns=1, timeout=0.1)
                if len(learner_ret_ref) > 0:
                    log_tuple, actor_grad, critic_grad = ray.get(learner_ret_ref)[0]
                    actor_gradients.append(actor_grad)
                    critic_gradients.append(critic_grad)
                    obj_c += log_tuple[0]
                    obj_a += log_tuple[1]
            obj_c /= num_nodes
            obj_a /= num_nodes
            # summed gradients
            summed_actor_gradients = [
                np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*actor_gradients)
            ]
            summed_critic_gradients = [
                np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*critic_gradients)
            ]
            # learners obtain global gradients and update weights
            for node in range(num_nodes):
                learners[node].set_gradients_and_update.remote(summed_actor_gradients, summed_critic_gradients, total_steps)
        print("time cost: ", time.time() - front_time, "s")

        # get current weights from learners
        actor_weights, critic_weights = ray.get(learners[0].get_weights.remote())
        # initialize evaluator
        if eval_run_ref is None:  # Evaluate the first time 
            eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, (obj_c, obj_a))]
        else:
            return_ref, eval_run_ref = ray.wait(object_refs=eval_run_ref, num_returns=1, timeout=0.1)
            if len(return_ref):  # if evaluator.run is done
                [if_Train, ref_list] = ray.get(return_ref)[0]
                if len(ref_list) > 0:
                    actor = ray.get(ref_list[0])
                    critic = ray.get(ref_list[1])
                    shared_net = ray.get(ref_list[2])
                    recorder = ray.get(ref_list[3])
                    torch.save(actor, cwd + '/actor.pth')
                    torch.save(critic, cwd + '/critic.pth')
                    torch.save(shared_net, cwd + '/shared_net.pth')
                    np.save(cwd + '/recorder.npy', recorder)
                    draw_learning_curve(recorder=np.array(recorder), save_path=cwd + '/learning_curve.jpg')
                eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, log_tuple)]
    
    recorder = ray.get(evaluator.get_recorder.remote())
    np.save(cwd + '/recorder.npy', recorder)
    draw_learning_curve(recorder=np.array(recorder), save_path=cwd + '/learning_curve.jpg')
 
    ref_list = ray.get(learners[1].save.remote())
    actor = ray.get(ref_list[0])
    critic = ray.get(ref_list[1])
    shared_net = ray.get(ref_list[2])    
    torch.save(actor, cwd + '/actor_final.pth')
    torch.save(critic, cwd + '/critic_final.pth')
    torch.save(shared_net, cwd + '/shared_net_final.pth')


if __name__ == '__main__':
    agent_class = PPO_discrete  # PPO_continuous/discrete, TD3, SAC
    env_class = chase3D
    args = ppo_args

    args.env_class = env_class
    args.agent_class = agent_class
    
    current_cwd = os.getcwd()
    print(current_cwd)
    train_agent_multiprocessing(args)
    print(current_cwd)
