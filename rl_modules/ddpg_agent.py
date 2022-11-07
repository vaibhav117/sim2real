import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.asym_models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from collections import deque
import wandb
import logger
from mpi_utils.mpi_moments import mpi_moments
"""
ddpg with HER (MPI-version)

"""
def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch



def dims_to_shapes(input_dims):
    return {key: tuple([val]) if isinstance(val, int) > 0  else tuple(val) for key, val in input_dims.items() }


class ddpg_agent:
    def __init__(self, args, env, env_params):
        
        self.args = args
        self.env = env
        self.env_params = env_params
        self.dimo = self.env_params['obs']
        self.dimg = self.env_params['g']
        self.dimu = self.env_params['action']
        self.dimo_image = self.env_params['obs_image']
        
        assert self.args.max_timesteps > 0
        # create the network
        self.actor_network = actor(env_params,self.args.action_max, self.args.feature_dim)
        self.critic_network = critic(env_params, self.args.action_max)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params, self.args.action_max,self.args.feature_dim)
        self.critic_target_network = critic(env_params, self.args.action_max)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.input_shapes = dims_to_shapes(self.env_params)
        buffer_shapes = {key: (self.args.max_timesteps-1 if key != 'obs' else self.args.max_timesteps, *self.input_shapes[key])
                         for key, val in self.input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.args.max_timesteps, self.dimg)
        buffer_shapes['obs_image'] = (self.args.max_timesteps,*self.dimo_image )
        # TODO T vs T-1 for obs_image-- use T because obs_image_next is needed
        buffer_size = (self.args.buffer_size // self.args.num_rollouts_per_mpi) * self.args.num_rollouts_per_mpi


        self.buffer = replay_buffer(buffer_shapes, buffer_size,self.args.max_timesteps, self.her_module.sample_her_transitions)
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['g'], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir,"model")
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        self.success_history = deque(maxlen=100)
        self.n_episodes = 0 

    # def learn(self):
    #     """
    #     train the network

    #     """
    #     # start to collect samples
    #     for epoch in range(self.args.n_epochs):
    #         for _ in range(self.args.n_cycles):
    #             mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
    #             for _ in range(self.args.num_rollouts_per_mpi):
    #                 # reset the rollouts
    #                 ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
    #                 # reset the environment
    #                 observation = self.env.reset()
    #                 obs = observation['observation']
    #                 ag = observation['achieved_goal']
    #                 g = observation['desired_goal']
    #                 obs_image = observation['image_observation']
    #                 # start to collect samples
    #                 for t in range(self.args['max_timesteps']):
    #                     with torch.no_grad():
    #                         input_tensor = self._preproc_inputs(obs, g, obs_image)
    #                         pi = self.actor_network(input_tensor)
    #                         action = self._select_actions(pi)
    #                     # feed the actions into the environment
    #                     observation_new, _, _, info = self.env.step(action)
    #                     obs_new = observation_new['observation']
    #                     ag_new = observation_new['achieved_goal']
    #                     obs_image_new = observation_new['image_observation']
    #                     # append rollouts
    #                     ep_obs.append(obs.copy())
    #                     ep_ag.append(ag.copy())
    #                     ep_g.append(g.copy())
    #                     ep_actions.append(action.copy())
    #                     # re-assign the observation
    #                     obs = obs_new
    #                     ag = ag_new
    #                 ep_obs.append(obs.copy())
    #                 ep_ag.append(ag.copy())
    #                 mb_obs.append(ep_obs)
    #                 mb_ag.append(ep_ag)
    #                 mb_g.append(ep_g)
    #                 mb_actions.append(ep_actions)
    #             # convert them into arrays
    #             mb_obs = np.array(mb_obs)
    #             mb_ag = np.array(mb_ag)
    #             mb_g = np.array(mb_g)
    #             mb_actions = np.array(mb_actions)
    #             # store the episodes
    #             self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
    #             self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
    #             for _ in range(self.args.n_batches):
    #                 # train the network
    #                 self._update_network()
    #             # soft update
    #             self._soft_update_target_network(self.actor_target_network, self.actor_network)
    #             self._soft_update_target_network(self.critic_target_network, self.critic_network)
    #         # start to do the evaluation
    #         success_rate = self._eval_agent()
    #         if MPI.COMM_WORLD.Get_rank() == 0:
    #             print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
    #             torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
    #                         self.model_path + '/model.pt')

    def reset_all_rollouts(self):
        self.obs_dict = self.env.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']
        if self.g.ndim ==1 :
            self.g = self.g.reshape(1,-1)
        self.initial_o_image = self.obs_dict['image_observation']
   
    def get_actor_output(self,actor_net, obs, g, obs_image):
        input_obs_image, input_g = self._preproc_inputs(obs,g, obs_image)
        input_obs_image = input_obs_image.reshape(-1, *self.dimo_image)
        input_g = input_g.reshape(-1, self.dimg)
        input_obs_image = torch.tensor(input_obs_image, dtype=torch.float32)
        input_g = torch.tensor(input_g, dtype= torch.float32)
        pi= actor_net(input_obs_image, input_g)
        return pi


    def generate_rollouts(self):
        self.reset_all_rollouts()
        obs = np.empty((self.args.num_rollouts_per_mpi, self.env_params['obs']), np.float32)
        ag = np.empty((self.args.num_rollouts_per_mpi, self.env_params['g']), np.float32)
        obs_image = np.empty((self.args.num_rollouts_per_mpi, *self.env_params['obs_image']), np.float32)
        obs[:] = self.initial_o
        ag[:] = self.initial_ag
        obs_image[:] = self.initial_o_image
        
        # generate episodes
        self.info_keys = [key.replace('info_', '') for key in self.env_params.keys() if key.startswith('info_')]
        
        ep_obs, ep_ag, ep_g, ep_o_image, ep_action, successes ,dones= [],[], [],[],[], [],[]
        
        info_values = [np.empty((self.args.max_timesteps - 1, self.args.num_rollouts_per_mpi, self.env_params['info_' + key]), np.float32) for key in self.info_keys]

        for t in range(self.args.max_timesteps):
            with torch.no_grad():
                pi = self.get_actor_output(self.actor_network,obs ,self.g, obs_image)
                action = self._select_actions(pi)
                # The non-batched case should still have a reasonable shape.

            obs_new = np.empty((self.args.num_rollouts_per_mpi, self.env_params['obs']),np.float32)
            ag_new = np.empty((self.args.num_rollouts_per_mpi, self.env_params['g']), np.float32)
            obs_image_new = np.empty((self.args.num_rollouts_per_mpi, *self.env_params['obs_image']), np.float32)
            success = np.zeros(self.args.num_rollouts_per_mpi)

            # feed the actions into the environment
            observation_dict_new,_ , done, info = self.env.step(action)
            obs_new = observation_dict_new['observation']
            ag_new = observation_dict_new['achieved_goal']
            obs_image_new = observation_dict_new['image_observation']
            success = np.array([info['is_success']])
            if done:
            # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
            # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
            # after a reset
                break

            # for i, info_dict in enumerate(info):
            for idx, key in enumerate(self.info_keys):
                info_values[idx][t,0] = info[key]

            if np.isnan(obs_new).any():
                self.logger.warn("Nan caught during rollout generation,..Trying again ..")
                self.reset_all_rollouts()
                return self.generate_rollouts()

            if action.ndim == 1:
                action = action.reshape(1,-1)
            dones.append(done)
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            successes.append(success.copy())
            ep_action.append(action.copy())
            ep_g.append(self.g.copy())
            ep_o_image.append(obs_image.copy())

            obs[...] = obs_new
            ag[...] = ag_new
            obs_image[...] = obs_image_new

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_o_image.append(obs_image.copy()) # TODO  Use this for goal image
        
        episode = dict(obs=ep_obs,
                       action=ep_action,
                       g=ep_g,
                       ag=ep_ag, obs_image=ep_o_image)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value




        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.args.num_rollouts_per_mpi,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        self.n_episodes += self.args.num_rollouts_per_mpi

        return convert_episode_to_batch_major(episode) # TODO check if this is neccessary




    def learn1(self):
        rank = MPI.COMM_WORLD.Get_rank()
        best_success_rate = -1
        self.args.n_epochs= self.args.total_timesteps // self.args.n_cycles // self.args.max_timesteps // self.args.num_rollouts_per_mpi

        for epoch in range(self.args.n_epochs):
            self.clear_history()
            for _ in range(self.args.n_cycles):
                episode= self.generate_rollouts()
                self.store_episode(episode)
            for _ in range(self.args.n_batches):
                self._update_network1()
            self._soft_update_target_network(self.actor_target_network, self.actor_network)
            self._soft_update_target_network(self.critic_target_network,self.critic_network)
            success_rate = self._eval_agent()
            wandb.log({"success_Rate": success_rate,'epoch':epoch})
            if rank == 0 and success_rate >= best_success_rate :
                print('[{}] epoch is: {} / {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, self.args.n_epochs, success_rate))
                torch.save([self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')

                logger.record_tabular("test/success_rate",success_rate)
                logger.record_tabular("current_episode_size", self.buffer.get_current_episode_size())
                logger.record_tabular("transitions_stored", self.buffer.get_transitions_stored())
                logger.record_tabular("current_size", self.buffer.get_current_size())
            
            for key, val in self.logs('train'):
                logger.record_tabular(key, mpi_average(val))
        
            if rank == 0:
                logger.dump_tabular()




    def store_episode(self, episode_batch, update_stats=True):
        self.buffer.store_episode(episode_batch)
        if update_stats:
            episode_batch['obs_next'] = episode_batch['obs'][:,1:,:]
            episode_batch['ag_next'] = episode_batch['ag'][:,1:,:]
            num_normalizing_transitions= episode_batch['action'].shape[0] * episode_batch['action'].shape[1]
            transitions= self.her_module.sample_her_transitions(episode_batch, num_normalizing_transitions)
            obs,g = transitions['obs'], transitions['g']
            transitions['obs'], transitions['g'] = self._preproc_og(obs,g)

            self.o_norm.update(transitions['obs'])
            self.g_norm.update(transitions['g'])
            # recompute the stats
            self.o_norm.recompute_stats()
            self.g_norm.recompute_stats()



    # pre_process the inputs
    def _preproc_inputs(self, obs, g, obs_image):
        obs_norm = self.o_norm.normalize(obs)       # TODO: Can be removed as it's not really being used
        g_norm = self.g_norm.normalize(g)
        if self.args.cuda:
            obs_image = torch.from_numpy(obs_image).cuda()
            g_norm = torch.from_numpy(g_norm).cuda()
        return obs_image, g_norm
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.args.action_max * np.random.randn(*action.shape)
        action = np.clip(action, -self.args.action_max, self.args.action_max)
        # random actions...
        random_actions = np.random.uniform(low=-self.args.action_max, high=self.args.action_max, \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'action': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    # def _update_network(self):
    #     # sample the episodes
    #     transitions = self.buffer.sample(self.args.batch_size)
    #     # pre-process the observation and goal
    #     o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
    #     transitions['obs'], transitions['g'] = self._preproc_og(o, g)
    #     transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
    #     # start to do the update
    #     obs_norm = self.o_norm.normalize(transitions['obs'])
    #     g_norm = self.g_norm.normalize(transitions['g'])
    #     inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
    #     obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
    #     g_next_norm = self.g_norm.normalize(transitions['g_next'])
    #     inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
    #     # transfer them into the tensor
    #     inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    #     inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
    #     actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
    #     r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
    #     if self.args.cuda:
    #         inputs_norm_tensor = inputs_norm_tensor.cuda()
    #         inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
    #         actions_tensor = actions_tensor.cuda()
    #         r_tensor = r_tensor.cuda()
    #     # calculate the target Q value function
    #     with torch.no_grad():
    #         # do the normalization
    #         # concatenate the stuffs
    #         actions_next = self.actor_target_network(inputs_next_norm_tensor)
    #         q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
    #         q_next_value = q_next_value.detach()
    #         target_q_value = r_tensor + self.args.gamma * q_next_value
    #         target_q_value = target_q_value.detach()
    #         # clip the q value
    #         clip_return = 1 / (1 - self.args.gamma)
    #         target_q_value = torch.clamp(target_q_value, -clip_return, 0)
    #     # the q loss
    #     real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
    #     critic_loss = (target_q_value - real_q_value).pow(2).mean()
    #     # the actor loss
    #     actions_real = self.actor_network(inputs_norm_tensor)
    #     actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
    #     actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
    #     # start to update the network
    #     self.actor_optim.zero_grad()
    #     actor_loss.backward()
    #     sync_grads(self.actor_network)
    #     self.actor_optim.step()
    #     # update the critic_network
    #     self.critic_optim.zero_grad()
    #     critic_loss.backward()
    #     sync_grads(self.critic_network)
    #     self.critic_optim.step()




    def _update_network1(self):
        transitions = self.buffer.sample(self.args.batch_size)
        o, o_next, g , o_image,o_image_next = transitions['obs'], transitions['obs_next'], transitions['g'] , transitions['obs_image'], transitions['obs_image_next']
        transitions['obs'], transitions['g'] = self._preproc_og(o,g)
        transitions['obs_next'] , transitions['g_next'] =  self._preproc_og(o_next, g)
        
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])

        # for critic
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # make tensor 
        inputs_norm_tensor = torch.tensor(inputs_norm,dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['action'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        g_norm_tensor = torch.tensor(g_norm,dtype=torch.float32)
        g_next_norm_tensor = torch.tensor(g_next_norm, dtype=torch.float32)
        o_image_tensor = torch.tensor(o_image, dtype=torch.float32)
        o_image_next_tensor = torch.tensor(o_image_next, dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
            g_norm_tensor = g_norm_tensor.cuda()
            g_next_norm_tensor = g_next_norm_tensor.cuda()
            o_image_tensor = o_image_tensor.cuda()
            o_image_next_tensor = o_image_next_tensor.cuda()

        with torch.no_grad():
            o_image_tensor = o_image_tensor.reshape(-1,*self.dimo_image)
            g_norm_tensor = g_norm_tensor.reshape(-1, self.dimg)
            actions_next = self.actor_target_network(o_image_next_tensor, g_next_norm_tensor)  #TODO change this to next tensors 
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        
        # the actor loss
        actions_real = self.actor_network(o_image_tensor, g_norm_tensor)
        actor_loss = - self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss+= self.args.action_l2 * (actions_real / self.args.action_max).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()


    

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            obs_image = observation['image_observation']
            for _ in range(self.args.max_timesteps):
                with torch.no_grad():
                    pi = self.get_actor_output(self.actor_network,obs, g, obs_image)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                obs_image = observation_new['image_observation']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        # self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)
    
    def logs(self, prefix="worker"):
        logs=[]
        logs+= [('success_rate', np.mean(self.success_history))]
        logs+= [('episode', self.n_episodes)]

        logs += [('stats_o/mean', np.mean([self.o_norm.mean]))]
        logs += [('stats_o/std', np.mean([self.o_norm.std]))]
        logs += [('stats_g/mean', np.mean([self.g_norm.mean]))]
        logs += [('stats_g/std', np.mean([self.g_norm.std]))]



        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
