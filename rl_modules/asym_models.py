import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define th=e actor network
class actor(nn.Module):
    def __init__(self, env_params,action_max, feature_dim):
        super(actor, self).__init__()
        self.max_action = action_max
        self.fc1 = nn.Linear(feature_dim + 10*env_params['g'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        self.feature_extractor = feature_extractor(env_params, feature_dim=feature_dim)
        self.apply(weight_init)
    def forward(self, obs_image, g):
        pi_obs_image = self.feature_extractor(obs_image)
        g_pi = torch.tile(g, (1,10))
        x = torch.cat((pi_obs_image, g_pi), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params, action_max):
        super(critic, self).__init__()
        self.max_action = action_max
        self.fc1 = nn.Linear(env_params['obs'] + env_params['g'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.apply(weight_init)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


#TODO feature extractor

class feature_extractor(nn.Module):
    def __init__(self, env_params, feature_dim=100):
        super(feature_extractor, self).__init__()
        obs_shape= env_params['obs_image'] #channel, height, width

        # assert obs_shape[2] ==3 or obs_shape[2] ==4
        
        self.feature_dim = feature_dim
        
        self.conv = nn.Sequential(
                                    nn.Conv2d(obs_shape[2], 32,kernel_size=8, stride=4),\
                                    nn.ReLU(), nn.Conv2d(32,64,kernel_size=4,stride=2),\
                                    nn.ReLU(), nn.Conv2d(64,64,kernel_size=3, stride=1),nn.ReLU(),
                                    nn.Flatten()
                                )

        conv_out_size = self._get_conv_out(obs_shape)   
        
        self.fc = nn.Sequential(
                                    nn.Linear(conv_out_size, 512), 
                                    nn.ReLU(),
                                    nn.Linear(512,self.feature_dim)
                                )

        
        self.apply(weight_init)

    def _get_conv_out(self, shape):
        torch_shape = (shape[2] , shape[0] , shape[1])
        o = self.conv(torch.zeros(1, *torch_shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # assert len(x.shape) == 4
        x= x.permute(0,3,1,2) # batch , channel, height, width
        conv_out = self.conv(x)
        return self.fc(conv_out)











def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)