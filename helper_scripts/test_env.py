import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from gym import envs
import gym
import gym_point
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
from video import VideoRecorder
import cv2


print("Hello")
# env1 = gym.make('XarmImageReach-v1', texture_rand=0, camera_rand=1, light_rand=1, depth=False, view=230)
# env2 = gym.make('XarmImagePickandPlace-v1', texture_rand=0, camera_rand=0, light_rand=0, depth=True, depth_noise=True,  view=215, crop_amount=16, action_variability=0)
env3 = gym.make('XarmImagePegInsertionEnv-v1', texture_rand=0, camera_rand=0, light_rand=0, depth=True, depth_noise=True,  view=215, crop_amount=16, action_variability=0)

# env1 = gym.make('point-v0'')
# env3 = gym.make('XarmImageReach-v1')
# env2 = gym.make('XarmImageReach-v1')
# env2 = gym.make('FetchImagePickandPlace-v1')


v = VideoRecorder(video_dir="video/test_env")
v.init(enabled=True)

env = env3
env._max_episode_steps = 50
env.reset()



def random_movement():
    success_count = 0
    for _ in range(100000):
        import time
        action = env.action_space.sample()

        obs , reward , done , info = env.step(action)
        print(f"obs:{obs}")
        print(f"reward:{reward} , info:{info} , done:{done}")
        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1
            env.reset()

        if done:
            env.reset()

        env.render(mode='human')

    print(success_count)

def random_movement_rgb():
    success_count = 0
    env.reset()
    for _ in range(100000):
        action = env.action_space.sample()
        action = [0.5,0.,-0.5,-1]
        print(action)
        obs , reward , done , info = env.step(action)
        print(f"reward:{reward} , info:{info} , done:{done}")
        if info['is_success']:
            success_count += 1
            # env.reset()

        if done:
            print(f"DONE")
            env.reset()

        f, axarr = plt.subplots(1,2,figsize=(7,5))
        
        axarr[0].imshow(obs["image_observation"][:,:,:3])
        axarr[1].imshow(obs["image_observation"][:,:,3])
        plt.show()

    print(success_count)

def save_video():
    success_count = 0
    for _ in range(250):
        import time
        action = env.action_space.sample()

        obs , reward , done , info = env.step(action)
        v.record(env)

        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1

        if done:
            env.reset()

        print(f"size:{obs['image_observation'].shape}")
    v.save("randomisations.mp4")
    print(success_count)


def scripted_action_reach():
    success_count = 0
    obs = env.reset()
    for _ in range(100000):
        import time
        # time.sleep(0.5)
        action = env.action_space.sample()
        action[:3] = obs['desired_goal'] - obs['achieved_goal']
        obs , reward , done , info = env.step(action)
        print(f"reward:{reward} , info:{info} , done:{done}")
        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1
            env.reset()

        if done:
            env.reset()

        env.render(mode='human')

    print(success_count)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)



def check_is_above(block_pos,gripper_pos):
    if abs(block_pos[0]-gripper_pos[0])<0.01 and abs(block_pos[0]-gripper_pos[0])<0.01 and abs(block_pos[0]-gripper_pos[0]) < 0.06:
        return True

def test_pick_and_place():
    success_count = 0
    episode_count = 0
    obs = env.reset()

    GRIPPER_CLOSE = 1
    GRIPPER_OPEN = -1
    gripper_state = GRIPPER_OPEN
    goal_pos = obs['observation'][3:6]

    is_above = False
    picked = False
    goal_reached = False

    while episode_count < 20:

        if not is_above and check_is_above(obs['observation'][3:6],obs['observation'][:3]):
            is_above = True

        action = env.action_space.sample()

        if not goal_reached:
            if not is_above:
                action[:3] = 20*(obs['observation'][3:6] - ( obs['observation'][:3] + np.array([0,-0.05,-0.05]) ))
            elif not picked:
                for _ in range(5):
                    action[:3] = [0,0,-4] 
                    env.step(action)
                    v.record(env)
                    # env.render(mode='human')
                for _ in range(5):
                    gripper_state = GRIPPER_CLOSE
                    action = [0,0,0,GRIPPER_CLOSE]
                    env.step(action)
                    v.record(env)
                    # env.render(mode='human')
                picked = True
            else:
                action[:3] = 20*(obs['desired_goal']-obs['achieved_goal'])
                gripper_state = GRIPPER_CLOSE
        else:
            action[:3] = [0,0,0]
            gripper_state = GRIPPER_CLOSE

        action[3] = gripper_state
        obs , reward , done , info = env.step(action)

        if info['is_success']:
            success_count += 1
            goal_reached = True
            # obs = env.reset()
            gripper_state=GRIPPER_OPEN
            goal_pos = obs['observation'][3:6]
            is_above = False
            picked = False

        if done:
            obs = env.reset()
            goal_reached = False
            gripper_state=GRIPPER_OPEN
            goal_pos = obs['observation'][3:6]
            is_above = False
            picked = False
            episode_count += 1

        v.record(env)
        # env.render(mode='human')

    v.save("pick_and_place_2.mp4")
    print(success_count)


def test_peg_insertion():
    success_count = 0
    env.reset()
    obs = env.reset()
    is_above = False

    for _ in range(100000):
        action = env.action_space.sample()

        print(f"desired_goal:{obs['desired_goal']} , slot_pos:{obs['observation']}")

        if not check_is_above(obs['observation'][:3],obs['desired_goal']):
            action[:3] = 20*(( obs['desired_goal'] + np.array([0,0,0.2]) - obs['observation'][:3]  ))
        else:
            action  = [0,0,-1,0]
        action[3] = -1
        print(action)
        obs , reward , done , info = env.step(action)

        print(f"reward:{reward} , info:{info} , done:{done}")
        if info['is_success']:
            success_count += 1
            # env.reset()

        if done:
            print(f"DONE")
            env.reset()

        f, axarr = plt.subplots(1,2,figsize=(7,5))
        
        axarr[0].imshow(obs["image_observation"][:,:,:3])
        axarr[1].imshow(obs["image_observation"][:,:,3])
        plt.show()

    print(success_count)

def test_pick_and_place_2():
    success_count = 0
    obs = env.reset()
    GRIPPER_CLOSE=1
    GRIPPER_OPEN=-1
    vertical_motion=-0.1
    gripper_state=GRIPPER_OPEN
    reached = False
    step_counter = 0
    for _ in range(100000):
        import time
        # time.sleep(0.5)
        action = env.action_space.sample()
        action = [0,0,vertical_motion,gripper_state]
        step_counter += 1
        obs , reward , done , info = env.step(action)
        print(f"gripper_pos:{obs['observation'][:3]}")
        print(f"step_counter:{step_counter}")
        if not reached:
            if obs['observation'][2]-obs['observation'][5] < -0.01:
                gripper_state = GRIPPER_CLOSE
                reached=True
                vertical_motion=0.4
                step_counter = 0

        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1
            env.reset()
            gripper_state=GRIPPER_OPEN
            goal_pos = obs['observation'][3:6]

        if done:
            env.reset()
            gripper_state=GRIPPER_OPEN
            goal_pos = obs['observation'][3:6]

        env.render(mode='human')

    print(success_count)

def test_motion():
    episodes = 0
    gripper_state = 1
    grip_change_counter=0
    GRIP_CHANGE_LIMIT=20
    while True:
        grip_change_counter += 1
        action = [0,0,0,1]
        # if grip_change_counter < GRIP_CHANGE_LIMIT:
        #     action = np.array([0.03,0.,0.1,1])
        # else:
        #     action = np.array([0.03,0.,0.1,1])

        obs , reward , done , info = env.step(action)
        v.record_obs(obs['image_observation'])
        # v.record(env)

        # plt.imshow(obs['image_observation'])
        # plt.show()
        # cv2.imshow("image",obs['image_observation'])
        # cv2.waitKey(0)
        
        # print(obs['observation'])

        if info['is_success']:
            episodes += 1

        if done:
            episodes += 1
            grip_change_counter = 0
            env.reset()

        if episodes>=1:
            break

    v.save("test_motion_test.mp4")

def test_env_limits():
    episodes = 0
    while True:
        action = np.array([1.,1.0,.0,.0]) 
        obs , reward , done , info = env.step(action)
        print(f"current gripper pos:{obs['observation'][:3]}")
        print(f"goal:{obs['desired_goal']}")
        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1
            episodes += 1
            env.reset()

        if done:
            episodes += 1
            env.reset()

        if episodes>=6:
            break

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(obs["image_observation"])
        axarr[0,1].imshow(obs["desired_goal_image"])
        plt.show()
        

def test_contacts():
    success_count = 0
    episodes = 0

    A = []
    while True:
        import time
        env.render(mode='human')
        action = np.array([.0,0.0,-0.1,.2]) 
        # action = env.action_space.sample()
        obs , reward , done , info = env.step(action)
        # print(f"\n\nreward:{reward}\n\n")
        c_list = np.array([i.dist for i in env.sim.data.contact])
        # print(f"\ncontact array min : {c_list.min()} , length: {len(c_list)} \n")
        # print(f"\ncontact array: {c_list[:7]}\n")

        print(f"active_contacts:{env.sim.data.active_contacts_efc_pos.min()}")
        plt.plot(np.array([i.dist for i in env.sim.data.contact]))
        # plt.show()
        if info['is_success']:
            print(f"\n\ncompletion reward:{reward}\n\n")
            success_count += 1
            episodes += 1
            env.reset()

        if done:
            episodes += 1
            env.reset()

        if episodes>=6:
            break



    print(success_count)


# test_pick_and_place()
# test_motion()
# test_env_limits()
test_peg_insertion()


# test_pick_and_place_2()

# random_movement()
# random_movement_rgb()

# save_video()
# random_movement_rgbd()

# scripted_action_reach()

# test_contacts()