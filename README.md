# distributed-reinforcement-learning
A distributed framework for single-agent/multi-agent reinforcement learning algorithm, developed by using ray cluster.

# quick start
1. Deploy the repo in two servers.
2. Launch ray cluster in head server:
   ray start --head --port=6388 --resources='{"node_0": 1}'
3. Launch ray cluster in the other servers:
   ray start --address='xxx.xx.xxx.xxx:6388' --resources='{"node_1": 1}'
4. Summit jobs
   ray job submit --working-dir ./ -- python main.py
