import torch
from rl_modules.asym_models import actor
from arguments import get_args
from create_env import create_env
import numpy as np
from video import VideoRecorder
import numpy as np
import wandb
import os
from typing import Dict, Iterable, Optional, Union
import glob
from datetime import datetime
from pytz import timezone

def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

# process the inputs
def process_inputs(o_image, o, g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_image_tensor = torch.tensor(o_image, dtype=torch.float32).reshape(1,*o_image.shape)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    g_tensor = torch.tensor(g_norm, dtype=torch.float32).reshape(1,*g_norm.shape)

    return o_image_tensor, g_tensor

    # TODO reshape to batch, dim i.e (1, *obs_clip.shape)



if __name__ == '__main__':
    args = get_args()
    global ts
    global index
    now_asia = datetime.now(timezone(args.time_location))
    format = "%m-%d-%H:%M"
    ts = now_asia.strftime(format)
    global_dir = os.path.abspath(__file__+args.global_file_loc)
    if args.index:
        print(args.index)
        index = args.index
    else:
        index = get_latest_run_id(global_dir, args.experiment_name)
    # load the model param
    wandb.init(project=f"Demo_{args.project_name}" , name=f"{args.experiment_name}_index={index}_depth={args.depth}_depth_noise={args.depth_noise}_action_l2={args.action_l2}_{args.reward_type}_batch_size={args.batch_size}_lr={args.lr_actor}_texture_rand={args.texture_rand}_camera_rand={args.camera_rand}_light_rand={args.light_rand}_crop={args.crop_amount}", tags=["demo",f"{args.env_name}"])
    args.save_dir = os.path.join(global_dir,args.experiment_name+'_'+str(index))
    model_path = os.path.join(args.save_dir , 'model/model.pt')
    g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = create_env()
    # get the env param
    observation = env.reset()
    # get the environment params

    env_params = {'obs': observation['observation'].shape[0],
            'g': observation['desired_goal'].shape[0],
            'obs_image': observation['image_observation'].shape,
            'action': env.action_space.shape[0],

            }
    action_max =env.action_space.high[0]
       # create the actor network
    actor_network = actor(env_params,action_max, args.feature_dim)
    actor_network.load_state_dict(model)
    actor_network.eval()
    v = VideoRecorder(video_dir=os.path.join(args.save_dir, "video"))
    v.init(enabled=True)
    num_steps = args.demo_length * env._max_episode_steps
    rewards =[]
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        obs_image = observation['image_observation']
        episode_reward = 0
        for t in range(env._max_episode_steps):
            # env.render()
            inputs = process_inputs(obs_image,obs, g, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(*inputs) # obs_image, g
            action = pi.detach().numpy().squeeze()

            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            # v.record_obs(observation_new['image_observation'])
            v.record(env)
            episode_reward+= reward
            obs = observation_new['observation']
            obs_image = observation_new['image_observation']
        rewards.append(episode_reward)
        wandb.log({"episode_reward": episode_reward, 'success': info['is_success']})
        print('the episode is: {}, is success: {} episode reward: {}'.format(i, info['is_success'], episode_reward))
    video_path =os.path.join(args.save_dir,f"video/{args.experiment_name}_{num_steps}_withoutdomainrand.mp4") 
    v.save(video_path)
    wandb.save(video_path)
    print("Average reward on evaluation" , np.mean(np.array(rewards)))