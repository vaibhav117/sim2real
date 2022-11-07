import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

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
# from xarm_deploy.robot import XArm
# from xarm_deploy.camera import Camera

import pickle
import click

args = get_args()

pickle_file_name = "pickled_stuff/env_saved_obs"

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
        }

model_to_be_used = "kirby_pickandplace_depth_noise_every_render_domainrand_208_bigbox_withoutfoam_2"

def process_inputs(o_image, g, g_mean, g_std, args):
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_image_tensor = torch.tensor(o_image, dtype=torch.float32).reshape(1,*o_image.shape)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    g_tensor = torch.tensor(g_norm, dtype=torch.float32).reshape(1,*g_norm.shape)

    return o_image_tensor, g_tensor

def init():
    # env = gym.make(args.env_name, view=args.view, reward_type=args.reward_type, depth=args.depth, texture_rand=args.texture_rand, camera_rand=args.camera_rand, depth_noise=args.depth_noise, light_rand=args.light_rand, crop_amount=args.crop_amount)
    env = create_env()
    model_path = model_loc[model_to_be_used]
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

    return g_mean, g_std, model , env, actor_network

def collect_images(env):
    images_to_be_saved = []
    
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

            observation_new, reward, _, info = env.step(action)

            episode_reward += reward
            
            obs = observation_new['observation']
            obs_image = observation_new['image_observation']
            images_to_be_saved.append(obs_image)
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(images_to_be_saved, f)

def dist(img_a, img_b):
    vec_a = get_feature_vector(img_a)
    vec_b = get_feature_vector(img_b)
    assert vec_a.shape == vec_b.shape
    return np.linalg.norm(vec_a - vec_b, axis=-1)[0]

def view_collected_images():
    with open(pickle_file_name, 'rb') as f:
        saved_images = pickle.load(f)
    for image in saved_images:
        cv2.imshow("image_saved",image)
        cv2.waitKey(100)

def get_feature_vector(img):
    feature_vector = actor_network.feature_extractor(img).detach().numpy()
    return feature_vector

def get_saved_images():
    goal = [1.3658339632502665,0.786722720344865,0.6317115473081438]
    with open(pickle_file_name, 'rb') as f:
        saved_images = pickle.load(f)
    processed_images = []
    processed_goals = []

    for obs_image in saved_images:
        processed_image , processed_goal = process_inputs(obs_image, goal, g_mean, g_std, args)
        processed_images.append(processed_image)
        processed_goals.append(processed_goal)
    
    return processed_images , processed_goals

def fade_depth_to_zero(img):
    img_temp = img.clone().detach()
    img_temp[0][:,:,3] = torch.zeros(84,84)
    return img_temp

def fade_red_to_zero(img):
    img_temp = img.clone().detach()
    img_temp[0][:,:,0] = torch.zeros(84,84)
    return img_temp

def fade_colors_to_zero(img):
    img_temp = img.clone().detach()
    img_temp[0][:,:,0] = torch.zeros(84,84)
    img_temp[0][:,:,1] = torch.zeros(84,84)
    img_temp[0][:,:,2] = torch.zeros(84,84)
    return img_temp    

def avg(list):
    return sum(list)/len(list)

if __name__ == '__main__':
    number_of_episodes = 10
    g_mean, g_std, model , env, actor_network = init()
    # collect_images()
    # view_collected_images()

    images , goals = get_saved_images()

    # plt.imshow(images[0].squeeze())
    # plt.show()
    # plt.imshow(images[5].squeeze())
    # plt.show()

    print(f"Difference btw 2 images with depth:{dist(images[0] , images[5])}")
    print(f"Difference btw same image with depth: {dist(images[0] , images[0])}")

    depth_faded_img_0 = fade_depth_to_zero(images[0])
    depth_faded_img_5 = fade_depth_to_zero(images[5])
    # plt.imshow(depth_faded_img_0.squeeze())
    # plt.show()
    # plt.imshow(depth_faded_img_5.squeeze())
    # plt.show()

    print(f"Difference btw 2 images without depth:{dist(depth_faded_img_0,depth_faded_img_5)}")
    print(f"Difference btw same image without depth: {dist(depth_faded_img_0,depth_faded_img_0)}")

    red_faded_img_0 = fade_red_to_zero(images[0])
    red_faded_img_5 = fade_red_to_zero(images[5])
    # plt.imshow(red_faded_img_0.squeeze())
    # plt.show()
    # plt.imshow(red_faded_img_5.squeeze())
    # plt.show()

    print(f"Difference btw 2 images without red:{dist(red_faded_img_0,red_faded_img_5)}")
    print(f"Difference btw same image without red: {dist(red_faded_img_0,red_faded_img_0)}")

    color_faded_img_0 = fade_colors_to_zero(images[0])
    color_faded_img_5 = fade_colors_to_zero(images[5])
    # plt.imshow(color_faded_img_0.squeeze())
    # plt.show()
    # plt.imshow(color_faded_img_5.squeeze())
    # plt.show()

    print(f"Difference btw 2 images without color:{dist(color_faded_img_0,color_faded_img_5)}")
    print(f"Difference btw same image without color: {dist(color_faded_img_0,color_faded_img_0)}")

    dist_btw_same_images = []
    dist_btw_diff_images = []

    dist_btw_same_images_without_depth = []
    dist_btw_diff_images_without_depth = []

    dist_btw_same_images_without_red = []
    dist_btw_diff_images_without_red = []

    dist_btw_same_images_without_color = []
    dist_btw_diff_images_without_color = []

    dist_btw_same_images_withandwithout_depth = []
    dist_btw_same_images_withandwithout_red = []
    dist_btw_same_images_withandwithout_colors = []

    print("\n\n")

    for index1 , img1 in enumerate(images[:10]):
        for index2, img2 in enumerate(images[:10]):
            # print(f"{index1} {index2}")
            if index1 == index2:
                dist_btw_same_images.append( dist(img1,img2) )
                dist_btw_same_images_without_depth.append( dist(fade_depth_to_zero(img1),fade_depth_to_zero(img2)) )
                dist_btw_same_images_without_red.append( dist(fade_red_to_zero(img1),fade_red_to_zero(img2)) )   
                dist_btw_same_images_without_color.append( dist(fade_colors_to_zero(img1),fade_colors_to_zero(img2)) )
                dist_btw_same_images_withandwithout_depth.append(dist(img1,fade_depth_to_zero(img2)))
                dist_btw_same_images_withandwithout_red.append(dist(img1,fade_red_to_zero(img2)))
                dist_btw_same_images_withandwithout_colors.append(dist(img1,fade_colors_to_zero(img2)))
            else:
                dist_btw_diff_images.append( dist(img1,img2) )
                dist_btw_diff_images_without_depth.append( dist(fade_depth_to_zero(img1),fade_depth_to_zero(img2)) )
                dist_btw_diff_images_without_red.append( dist(fade_red_to_zero(img1),fade_red_to_zero(img2)) )   
                dist_btw_diff_images_without_color.append( dist(fade_colors_to_zero(img1),fade_colors_to_zero(img2)) )         
    
    print(f"Average dist_btw_same_images:{avg(dist_btw_same_images)}")
    print(f"Average dist_btw_same_images_without_depth:{avg(dist_btw_same_images_without_depth)}")
    print(f"Average dist_btw_same_images_without_red:{avg(dist_btw_same_images_without_red)}")
    print(f"Average dist_btw_same_images_without_color:{avg(dist_btw_same_images_without_color)}")

    print("\n\n")

    print(f"Average dist_btw_diff_images:{avg(dist_btw_diff_images)}")
    print(f"Average dist_btw_diff_images_without_depth:{avg(dist_btw_diff_images_without_depth)}")
    print(f"Average dist_btw_diff_images_without_red:{avg(dist_btw_diff_images_without_red)}")
    print(f"Average dist_btw_diff_images_without_color:{avg(dist_btw_diff_images_without_color)}")

    print("\n\n")

    print(f"Average dist_btw_same_images_withandwithout_depth:{avg(dist_btw_same_images_withandwithout_depth)}")
    print(f"Average dist_btw_same_images_withandwithout_red:{avg(dist_btw_same_images_withandwithout_red)}")
    print(f"Average dist_btw_same_images_withandwithout_colors:{avg(dist_btw_same_images_withandwithout_colors)}")

    # tmp_img_1 = images[0]
    # tmp_img_1[:,:,3] = np.
    # print(f"Difference btw 2 images without depth: {dist(images[0] , images[0])}")
    
    # import ipdb; ipdb.set_trace()


    
    

