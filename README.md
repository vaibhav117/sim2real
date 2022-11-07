My goal is to learn an RL policy on an image based task in simulation and successfully transfer it to the real world on hte robot Xarm.

I also do not use any real world demonstration or feedback to improve the policy or the algorithm.

The basic model is inspired by [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)



# Challanges
1. Firstly, learning policies even in simulation for tasks such as Pick and Place because of their sparse rewards is itself quite challnaging.

2. 

# Approach
1. Use Asym Actor-Critic approach to learn a policy in a Mujoco environment that is made to look like our real setup.

2. Use Domain Randomization during the learing process to make the policy more resilient to changes in variations in Visual features such as lighting and camera angles.

3. Use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) to allow more efficient learning in a Sparse Reward setting.


# Results

Pick & Place in simulation | Pick & Place on the Real Robot 
-----------------------|-----------------------|
![](figures/sim2real_real_robot_deployment.gif)| ![](figures/sim2real_sim_deploy.gif)