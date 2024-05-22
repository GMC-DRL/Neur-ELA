import argparse
# import time


def get_config(args=None):
    parser = argparse.ArgumentParser()
    # Common config
    parser.add_argument('--dataset', default='bbob', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'],
                        help='specify the problem suite')
    parser.add_argument('--dim', type=int, default=10, help='dimension of search space')
    
    parser.add_argument('--device', default='cpu', help='device to use')    
    

    # Training parameters
    parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
    parser.add_argument('--max_learning_time', type=int, default=20 * 60, help='the maximum learning time for training, 30 min as default')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size of train set')
    parser.add_argument('--train_agent', default=None, help='agent for training')
    parser.add_argument('--train_optimizer', default=None, help='optimizer for training')
    parser.add_argument('--use_ela', default=False, type=bool, help='whether to use ela feature')
    parser.add_argument('--count_ela_fes', default=True, type=bool, help='whether to count the ela feature consuming fes')
    parser.add_argument('--hidden_dim', type=int, default=16, choices=[16, 64, 128])
    parser.add_argument('--n_layers', type=int, default=1, choices=[1, 3, 5])
    

    config = parser.parse_args(args)
    config.maxFEs = 2000 * config.dim
    
    config.n_logpoint = 50

    # config.save_interval = config.max_learning_step // config.n_checkpoint
    config.log_interval = config.maxFEs // config.n_logpoint

    return config
