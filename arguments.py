import argparse

"""
Here are the param for the training

"""

def get_args(s=None):
    if s is None:
        parser= argparse.ArgumentParser(s)
    else:
        parser = argparse.ArgumentParser()

    # the environment setting
    parser.add_argument('--project-name',type=str, default="pytoch-visual-learning")
    parser.add_argument('--experiment-name',type=str, default="pytorch_fetch_reach_asym")
    parser.add_argument('--env-name', type=str, default='FetchImageReach-v1', help='the environment name')
    parser.add_argument('--total-timesteps', type=int, default=2000000, help='total timesteps for training')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.3, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=8e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=8e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')
    parser.add_argument('--view', type=int, default=180, help='view of camera ')
    parser.add_argument('--reward-type', type=str, default="sparse", help='sparse or dense reward')
    parser.add_argument('--depth', action='store_true', help='use depth')
    parser.add_argument('--depth-noise', action='store_true', help='add noise to depth')
    parser.add_argument('--texture-rand', type=int, default=0, help='0:Disable , 1:Randomise only at reset , 2:Reset at every frame')
    parser.add_argument('--camera-rand', type=int, default=0, help='0:Disable , 1:Randomise only at reset , 2:Reset at every frame')
    parser.add_argument('--light-rand', type=int, default=0, help='0:Disable , 1:Randomise only at reset , 2:Reset at every frame')
    parser.add_argument('--feature-dim', type=int, default=100, help='encoder feature dim')
    parser.add_argument('--crop-amount', type=int, default=0, help='Number of pixels from all 4 sides to be cropped')
    parser.add_argument('--global-file-loc', default= "/../experiments", type=str)
    parser.add_argument('--time-location', default="Asia/Kolkata", type=str)
    parser.add_argument('--index', default=None, help="directory index ")
    parser.add_argument('--action-variability', type=float, default=0, help='Unpredictibility added to action_step')
    args = parser.parse_args()

    return args
