import pickle

def get_epoch_dict(dataset):
    epoch_choices = {
        'bbob': {
            'LDE': 70,
            'RL_PSO': 6,
            'RLEPSO': 175,
            'RL_DAS': 650,
            'DE_DDQN': 2,
            'GLEET': 62,
        },
        'bbob-noisy':{
            'LDE': 40,
            'RL_PSO': 3,
            'RLEPSO': 105,
            'RL_DAS': 400,
            'DE_DDQN': 1,
            'GLEET': 36,
        },
        'protein_docking':{
            'LDE': 1,  
            'RL_PSO': 6,    
            'RLEPSO': 24,  
            'RL_DAS': 28,  
            'DE_DDQN': 3,   
            'GLEET': 20,
        }

    }
    return epoch_choices.get(dataset)

def construct_problem_set(dataset):
    if dataset == 'bbob':
        with open('dataset/trainset_v2.pkl', 'rb') as f:
            trainset = pickle.load(f)
        with open('dataset/testset_v2.pkl', 'rb') as f:
            testset = pickle.load(f)
    elif dataset == 'protein_docking':
        with open('dataset/pd_trainset.pkl', 'rb') as f:
            trainset = pickle.load(f)
        with open('dataset/pd_testset.pkl', 'rb') as f:
            testset = pickle.load(f)
    elif dataset == 'bbob-noisy':
        with open('dataset/noisy_trainset.pkl', 'rb') as f:
            trainset = pickle.load(f)
        with open('dataset/noisy_testset.pkl', 'rb') as f:
            testset = pickle.load(f)
    return trainset,testset