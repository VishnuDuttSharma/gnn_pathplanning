"""
Data loader for coverage task
"""



import numpy as np
import random
import torch
import pickle

# Data shapes reference: https://github.com/proroklab/gnn_pathplanning/blob/master/dataloader/Dataloader_dcplocal_notTF_onlineExpert.py#L147-L149
class GNNCoverageDataset(Dataset):
    """Coverage task dataset, generated randomly and saved"""

    def __init__(self, datafile):
        """
        Parameters
        ----------
            datafile: Name of the dataset file
            
        """
        data = pickle.load(open(datafile, 'rb'))
        self.features = data[0]
        self.adj_mat  = data[1]
        self.targets  = data[2]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.adj_mat[idx]), torch.LongTensor(self.targets[idx])

        
def generate_data(data_size):
    feat_list, adj_list, label_list = [], [], []
    for _ in range(data_size):
        grid = get_reward_grid(height=HEIGHT, width=WIDTH, reward_thresh=REWARD_THRESH)
        robot_pos, adj_mat = get_initial_pose(grid, comm_range=20)

        cent_act, cent_rwd = centralized_greedy_action_finder(grid, robot_pos, fov=FOV)
        rand_act, rand_rwd = random_action_finder(grid, robot_pos, 1000)

        if cent_rwd > rand_rwd:
            action_vec = cent_act
        else:
            action_vec = rand_act
        
        feat_vec = get_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=NUM_TGT_FEAT, robot_feat_size=NUM_ROB_FEAT)

        feat_list.append(feat_vec)
        adj_list.append(adj_mat)
        
        action_one_hot = np.zeros((NUM_ROBOT, len(DIR_LIST)), dtype=np.uint8)
        action_one_hot[np.arange(NUM_ROBOT), action_vec] = 1
        label_list.append(action_one_hot)
    
    return [np.array(feat_list), np.array(adj_list), np.array(label_list)]


 
