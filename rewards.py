# import torch
# import sys

# def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
#     """
#     Compute diversity reward and representativeness reward

#     Args:
#         seq: sequence of features, shape (1, seq_len, dim)
#         actions: binary action sequence, shape (1, seq_len, 1)
#         ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
#         temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
#         use_gpu (bool): whether to use GPU
#     """
#     _seq = seq.detach()
#     _actions = actions.detach()
#     pick_idxs = _actions.squeeze().nonzero().squeeze()
#     num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
#     if num_picks == 0:
#         # give zero reward is no frames are selected
#         reward = torch.tensor(0.)
#         if use_gpu: reward = reward.cuda()
#         return reward

#     _seq = _seq.squeeze()
#     n = _seq.size(0)

#     # compute diversity reward
#     if num_picks == 1:
#         reward_div = torch.tensor(0.)
#         if use_gpu: reward_div = reward_div.cuda()
#     else:
#         normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
#         dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
#         dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
#         if ignore_far_sim:
#             # ignore temporally distant similarity
#             pick_mat = pick_idxs.expand(num_picks, num_picks)
#             temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
#             dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
#         reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

#     # compute representativeness reward
#     dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
#     dist_mat = dist_mat + dist_mat.t()
#     dist_mat.addmm_(1, -2, _seq, _seq.t())
#     dist_mat = dist_mat[:,pick_idxs]
#     dist_mat = dist_mat.min(1, keepdim=True)[0]
#     #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
#     reward_rep = torch.exp(-dist_mat.mean())

#     # combine the two rewards
#     reward = (reward_div + reward_rep) * 0.5

#     return reward
import torch

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False, coverage_weight=0.3, stability_weight=0.2):
    """
    Compute diversity, representativeness, coverage, and temporal stability rewards.

    Args:
        seq (torch.Tensor): sequence of features, shape (1, seq_len, dim)
        actions (torch.Tensor): binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU (default: False)
        coverage_weight (float): weighting factor for the coverage reward (default: 0.3)
        stability_weight (float): weighting factor for the temporal stability reward (default: 0.2)

    Returns:
        torch.Tensor: the combined reward value
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        return torch.tensor(0., device='cuda' if use_gpu else 'cpu')

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # Compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0., device='cuda' if use_gpu else 'cpu')
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())

    # Compute coverage reward
    sim_matrix = torch.matmul(normed_seq, normed_seq.t())
    max_sim, _ = sim_matrix[:, pick_idxs].max(dim=1)  # max similarity to any pick
    reward_coverage = max_sim.mean()  # average maximum similarity

    # Compute temporal stability reward
    if num_picks > 1:
        temporal_distances = torch.abs(pick_idxs[1:] - pick_idxs[:-1])
        stability_reward = torch.exp(-temporal_distances.float()).mean()
    else:
        stability_reward = torch.tensor(1.0, device='cuda' if use_gpu else 'cpu')  # Max stability if only one pick

    # Combine all rewards
    reward = (reward_div + reward_rep + coverage_weight * reward_coverage + stability_weight * stability_reward) / (3 + coverage_weight + stability_weight)

    return reward
