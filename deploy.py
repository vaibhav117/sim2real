from ast import arg
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2

from create_env import create_env
from rl_modules.asym_models import actor

from video import VideoRecorder
from arguments import get_args
from xarm_deploy.robot import XArm
from xarm_deploy.camera import Camera
from xarm_deploy.deploy_goals import env_goals

args = get_args()

coordinate_shift = [1.506,0.831,0.228]

model_loc = {
            "reach_everyepisode_domainrand_208":"/home/vaibhav/Projects/pytorch-visual-learning/saved_models/XarmImageReach-v1/with_domain_rand/kirby_xarm_reach_sideview208_everyrender_domainrand.pt",
            "pickandplace_domainrand_230":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_sim2real_domainrand.pt",
            "pickandplace_norand_view_230":"saved_models/XarmImagePickandPlace-v1/no_domain_rand/lambda_xarm_pickandplace_sim2real_norand.pt",
            "pickandplace_every_episode_domainrand_208":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_xarm_pickandplace_sideview-208_everyepisode_domainrand.pt",
            "pickandplace_every_render_domainrand_208":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_xarm_pickandplace_sideview-208_everyrender_domainrand.pt",

            # New models without the foam
            "kirby_pickandplace_depth_noise_every_render_domainrand_208_bigbox_withoutfoam":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_pickandplace_depth_noise_every_render_domainrand_208_bigbox_withoutfoam.pt",
            "kirby_pickandplace_depth_noise_every_render_domainrand_208_bigbox_withoutfoam_2":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_pickandplace_depth_noise_every_render_domainrand_208_bigbox_withoutfoam_2.pt",
            "xarm_pickandplace_depth_noise_every_render_domainrand_180_bigbox_withoutfoam":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/xarm_pickandplace_depth_noise_every_render_domainrand_180_bigbox_withoutfoam.pt",
            "lambda_xarm_pickandplace_table_sideview_depth_noise_real_constrains":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_sideview_depth_noise_real_constrains.pt",
            "lambda_xarm_pickandplace_table_sideview_depth_noise_blur":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_sideview_depth_noise_blur.pt",
            "lambda_xarm_pickandplace_table_sideview_depth_noise_blur_2":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_sideview_depth_noise_blur_2.pt",
            "lambda_xarm_pickandplace_table_sideview_depth_noise_blur_text_rand":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_sideview_depth_noise_blur_text_rand.pt",
            "lambda_xarm_pickandplace_table_sideview_depth_noise_blur_text_rand_2":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_sideview_depth_noise_blur_text_rand_2.pt",
            
            # Current Working Model
            "lambda_xarm_pickandplace_table_realenv":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_realenv.pt",
            
            "lambda_xarm_pickandplace_table_realenv_textrand":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_table_realenv_textrand.pt",
            
            # FInal check models
            "kirby_xarm_pickandplace_stable_test_airgoals":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_xarm_pickandplace_stable_test_airgoals.pt",
            "kirby_xarm_pickandplace_stable_test_airgoals_textrand":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/kirby_xarm_pickandplace_stable_test_airgoals_textrand.pt",
            "lambda_xarm_pickandplace_stable_test_airground_goals":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_stable_test_airground_goals.pt",
            "lambda_xarm_pickandplace_stable_test_airground_goals_actionvar":"saved_models/XarmImagePickandPlace-v1/with_domain_rand/lambda_xarm_pickandplace_stable_test_airground_goals_actionvar.pt",
        }

test_in_env = False  
model_to_be_used = "lambda_xarm_pickandplace_stable_test_airground_goals_actionvar"
enable_arm = True
enable_gripper = True

goals = env_goals[model_to_be_used]

def process_inputs(o_image, g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_image_tensor = torch.tensor(o_image, dtype=torch.float32).reshape(1,*o_image.shape)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    g_tensor = torch.tensor(g_norm, dtype=torch.float32).reshape(1,*g_norm.shape)

    return o_image_tensor, g_tensor

def process_real_world_inputs(o_image, g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_image_tensor = torch.tensor(o_image, dtype=torch.float32).reshape(1,*o_image.shape)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    g_tensor = torch.tensor(g_norm, dtype=torch.float32).reshape(1,*g_norm.shape)    
    return o_image_tensor, g_tensor

def init_arm():
    arm = XArm()
    arm.start_robot()
    arm.clear_errors()
    arm.set_mode_and_state()
    arm.reset(home=True)
    arm.goto_zero()
    time.sleep(2)
    return arm

def arm_refresh(arm):
    # arm.start_robot()
    arm.clear_errors()
    arm.set_mode_and_state()
    arm.reset(home=True)
    arm.goto_zero()
    time.sleep(2)

def inits():
    print(f"ENV_NAME: {args.env_name}")
    global enable_gripper
    env = create_env()
    model_path = model_loc[model_to_be_used]

    if args.env_name == "XarmImagePickandPlace-v1":
        enable_gripper = True

    g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    observation = env.reset()

    env_params = {
            'obs': observation['observation'].shape[0],
            'g': observation['desired_goal'].shape[0],
            'obs_image': observation['image_observation'].shape,
            'action': env.action_space.shape[0],
            }

    action_max = env.action_space.high[0]
    actor_network = actor(env_params,action_max, args.feature_dim)
    actor_network.load_state_dict(model)
    actor_network.eval()

    if test_in_env or not enable_arm:
        arm = None
    else:
        arm = init_arm()

    print(args.view)
    if args.view == 180:
        cam = Camera(view="front")
    else:
        cam = Camera(view="side")

    v = VideoRecorder(video_dir=os.path.join("video/deploy_script_generated/simulation", "env_frames"))
    v_frames = VideoRecorder(video_dir=os.path.join("video/deploy_script_generated/simulation", "obs_frames"))
    v.init(enabled=True)
    v_frames.init(enabled=True)

    return g_mean, g_std, model , env, actor_network, arm , cam , v, v_frames

def test_arm_movement(arm):
    print(f"Arm Pos:{arm.get_position()}")
    positions = [
                    [0,0,0],   
                    [1,1,1],
                    [-1,-1,1]       
                ]
    for pos in positions:
        scale_up_factor = 1
        pos = [element * scale_up_factor for element in pos]
        print(f"Going to: {pos}")
        arm.set_position(pos)
        input("Press Enter to continue...")

if __name__ == '__main__':    
    number_of_episodes = 20
    g_mean, g_std, model , env , actor_network , arm , cam , v , v_frames = inits()
    num_steps = number_of_episodes * env._max_episode_steps

    rewards =[]
    print(f"Starting deployment")

    # test_arm_movement(arm)

    if test_in_env:
        for i in range(number_of_episodes):
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            obs_image = observation['image_observation']
            episode_reward = 0
            for t in range(env._max_episode_steps):
                inputs = process_inputs(obs_image, g, g_mean, g_std, args)
                with torch.no_grad():
                    pi = actor_network(*inputs) # obs_image, g
                action = pi.detach().numpy().squeeze()
                
                # f, axarr = plt.subplots(1,2,figsize=(20,7))
                # axarr[0].imshow(inputs[0][0][:,:,:3])
                # axarr[1].imshow(inputs[0][0][:,:,3])
                # plt.show()

                observation_new, reward, _, info = env.step(action)
                v.record(env)
                v_frames.record_obs(observation_new['image_observation'])
                episode_reward += reward
                
                obs = observation_new['observation']
                obs_image = observation_new['image_observation']
            rewards.append(episode_reward)
            print(f"[{g[0]},{g[1]},{g[2]}],")
        v.save(f"{model_to_be_used}_view={args.view}_texturerand={args.texture_rand}_camerarand={args.camera_rand}_crop={args.crop_amount}_depth={args.depth}.mp4")
        v_frames.save(f"{model_to_be_used}_view={args.view}_texturerand={args.texture_rand}_camerarand={args.camera_rand}_crop={args.crop_amount}_depth={args.depth}.mp4")

    else:
        action_multiplier = 0.3

        # Take out first faulty iamge
        obs, image_frame, image = cam.get_frame()
        v_frames_goal = VideoRecorder(video_dir=os.path.join("video/deploy_script_generated/real_robot_recording/obs_frames", f"3_{model_to_be_used}_view={args.view}_texturerand={args.texture_rand}_camerarand={args.camera_rand}_crop={args.crop_amount}_depth={args.depth}"))

        for index , goal_current in enumerate(goals): # [[2.32,0.614,2]]:
            goal = goal_current
            # goal[2] += 0.4
            # goal[0] += 0.3
            print(f"goal:{goal}")
            for t in range(30):
                obs, image_frame, image = cam.get_frame()
                cv2.imshow("Frame",obs[:,:,:3])
                cv2.waitKey(100)
                
                inputs = process_real_world_inputs(obs, goal, g_mean, g_std, args)

                with torch.no_grad():
                    pi = actor_network(*inputs)

                action = (pi.detach().numpy().squeeze()) * action_multiplier
                print(f"Predicted action:{action}")
                
                f, axarr = plt.subplots(1,2,figsize=(10,5))
                axarr[0].imshow(inputs[0][0][:,:,:3])
                axarr[1].imshow(inputs[0][0][:,:,3])
                plt.show()

                for _ in range (10):
                    v_frames_goal.record_obs(inputs[0][0])

                if enable_arm:
                    new_pos = arm.get_position() + action[:3]
                    gripper_movement = 0
                    
                    #------ SAFETY CHECKS FOR ROBOT -------
                    if new_pos[2]<=-0.50 or new_pos[2]>3:
                        new_pos[2] = -0.54

                    # if new_pos[1]<-3.7 or new_pos[1]>4:
                    #     new_pos[1] -= action[1]

                    # if new_pos[0] < 0 or new_pos[0]>3:
                    #     new_pos[0] -= action[0]
                    #---------------------------------------
                    if enable_gripper:
                        if action[3] > 0:
                            gripper_movement = -200
                        else:
                            gripper_movement = 200
                        new_gripper_pose = arm.get_gripper_position() + gripper_movement
                        arm.set_gripper_position(new_gripper_pose)
                    print(f"current_pos:{arm.get_position()}")
                    arm.set_position(new_pos)   

                time.sleep(1)
            
            input("Press Enter to continue...")
            arm_refresh(arm)
            v_frames_goal.save(f"{index}.mp4")
                    
    
