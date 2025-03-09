import torch

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, 
                   use_gpu=False, coverage_weight=0.3, stability_weight=0.2):
    """
    Compute diversity, representativeness, coverage, and temporal stability rewards.

    Args:
        seq (torch.Tensor): sequence of features, shape (1, seq_len, dim)
        actions (torch.Tensor): binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity.
        temp_dist_thre (int): threshold for ignoring temporally distant similarity.
        use_gpu (bool): whether to use GPU.
        coverage_weight (float): weighting factor for the coverage reward.
        stability_weight (float): weighting factor for the temporal stability reward.

    Returns:
        torch.Tensor: the combined reward value.
    """
    _seq = seq
    _actions = actions

    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        return torch.tensor(0., device='cuda' if use_gpu else 'cpu')

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # Normalize sequence features
    normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)

    # Compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0., device='cuda' if use_gpu else 'cpu')
    else:
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # Dissimilarity matrix
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.  
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(_seq, _seq.t(), beta=1, alpha=-2)  # Fixed deprecated addmm_
    dist_mat = dist_mat[:, pick_idxs]
    min_dist = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-min_dist.mean())  

    # Compute coverage reward
    sim_matrix = torch.matmul(normed_seq, normed_seq.t())  
    max_sim, _ = sim_matrix[:, pick_idxs].max(dim=1)  
    reward_coverage = max_sim.mean()  

    # ✅ Fix: Ensure reward_stability is always defined
    if num_picks > 1:
        frame_diffs = torch.diff(pick_idxs.float())  
        frame_diffs = frame_diffs / (frame_diffs.max() + 1e-6)  # Avoid division by zero
        reward_stability = torch.exp(-frame_diffs.mean())  
    else:
        reward_stability = torch.tensor(1., device='cuda' if use_gpu else 'cpu')

    # Combine all rewards
    reward = (reward_div + reward_rep + coverage_weight * reward_coverage + 
              stability_weight * reward_stability) / (2 + coverage_weight + stability_weight)
    reward = reward.clone().detach().requires_grad_(True)  # Ensure it requires gradients
    return reward
