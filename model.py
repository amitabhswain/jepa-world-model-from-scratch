"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
import torch

def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    """
    Initialize a random agent position inside a square 2D room.

    Args:
        room_size: side length of the square grid
        seed: if not None, makes the sampling deterministic

    Returns:
        (2,) float32 tensor with integer-valued coordinates in [0, room_size - 1].
    """
    if seed is not None:
        torch.manual_seed(seed)

    pos = torch.randint(0, room_size, (2,))
    return pos.float()

# Step 2 - apply_action
import torch

def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    """
    Apply a discrete action to the agent state with wall clamping.

    Args:
        state: (2,) float tensor of (x, y) coordinates
        action: 0=up, 1=down, 2=left, 3=right
        room_size: side length of the square room

    Returns:
        New (2,) float tensor with coordinates clamped to [0, room_size - 1].
        The input state is not modified.
    """
    new_state = state.clone()

    if action == 0:      # up: y decreases
        new_state[1] -= 1
    elif action == 1:    # down: y increases
        new_state[1] += 1
    elif action == 2:    # left: x decreases
        new_state[0] -= 1
    elif action == 3:    # right: x increases
        new_state[0] += 1

    return torch.clamp(new_state, min=0, max=room_size - 1)

# Step 3 - render_observation
import torch

def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    """
    Render an agent state as a single-channel pixel observation.

    Args:
        state: (2,) float tensor of (x, y), x = column, y = row
        room_size: side length of the square room

    Returns:
        (1, room_size, room_size) float32 tensor, 0.0 background,
        1.0 at the agent pixel [0, y, x].
    """
    obs = torch.zeros(1, room_size, room_size, dtype=torch.float32)

    x = int(state[0].item())
    y = int(state[1].item())

    obs[0, y, x] = 1.0

    return obs

# Step 4 - env_reset
import torch

def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reset the 2D room environment.

    Args:
        room_size: side length of the square room
        seed: optional seed for reproducible initialization

    Returns:
        (state, observation): state is a (2,) float32 tensor of (x, y),
        observation is a (1, room_size, room_size) float32 image.
    """
    state = init_env_state(room_size=room_size, seed=seed)
    obs = render_observation(state, room_size=room_size)
    return state, obs

# Step 5 - env_step
import torch

def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Advance the 2D room by one discrete action.

    Args:
        state: (2,) float tensor of (x, y)
        action: 0=up, 1=down, 2=left, 3=right
        room_size: side length of the square room

    Returns:
        (next_state, next_obs): next_state is (2,) float32,
        next_obs is (1, room_size, room_size) float32.
    """
    next_state = apply_action(state, action, room_size=room_size)
    next_obs = render_observation(next_state, room_size=room_size)
    return next_state, next_obs

# Step 6 - collect_random_transitions
import torch

def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    """
    Collect (obs, action, next_obs, state, next_state) transitions by rolling
    out random actions in the 2D room, as one continuous trajectory.

    Args:
        num_transitions: number of transitions N to collect
        room_size: side length of the square room
        seed: RNG seed for reproducibility

    Returns:
        dict with 'observations' (N, 1, H, W), 'actions' (N,) long,
        'next_observations' (N, 1, H, W), 'states' (N, 2), 'next_states' (N, 2).
    """
    torch.manual_seed(seed)

    state, obs = env_reset(room_size=room_size, seed=seed)

    N = num_transitions
    observations = torch.zeros(N, 1, room_size, room_size, dtype=torch.float32)
    next_observations = torch.zeros(N, 1, room_size, room_size, dtype=torch.float32)
    actions = torch.zeros(N, dtype=torch.long)
    states = torch.zeros(N, 2, dtype=torch.float32)
    next_states = torch.zeros(N, 2, dtype=torch.float32)

    for i in range(N):
        action = int(torch.randint(0, 4, ()).item())

        next_state, next_obs = env_step(state, action, room_size=room_size)

        observations[i] = obs
        actions[i] = action
        next_observations[i] = next_obs
        states[i] = state
        next_states[i] = next_state

        state, obs = next_state, next_obs

    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'states': states,
        'next_states': next_states,
    }

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    """
    Build a JEPA training-ready transition dataset from random rollouts.

    Args:
        num_transitions: number of transitions N (default 512)
        room_size: side length of the square room (default 8)
        seed: RNG seed for reproducibility (default 0)

    Returns:
        dict with 'observations' (N, 1, H, W), 'actions' (N,) long,
        'next_observations' (N, 1, H, W), 'states' (N, 2), 'next_states' (N, 2).
    """
    return collect_random_transitions(
        num_transitions=num_transitions,
        room_size=room_size,
        seed=seed,
    )

# Step 8 - init_encoder_params
import torch

def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    """
    Initialize parameters of a small CNN encoder mapping
    (B, obs_channels, room_size, room_size) to (B, latent_dim).

    conv1: obs_channels -> 16, 3x3, stride 1, padding 1
    conv2: 16 -> 32,           3x3, stride 2, padding 1
    fc:    32*H*W -> latent_dim
    """
    torch.manual_seed(seed)

    # Spatial size after each conv: floor((H_in + 2p - k) / s) + 1, with k=3, p=1
    h1 = (room_size + 2 * 1 - 3) // 1 + 1   # conv1, stride 1
    h2 = (h1 + 2 * 1 - 3) // 2 + 1          # conv2, stride 2
    fc_in = 32 * h2 * h2

    params = {}
    params['conv1_w'] = (torch.randn(16, obs_channels, 3, 3) * 0.1).requires_grad_(True)
    params['conv1_b'] = torch.zeros(16, requires_grad=True)
    params['conv2_w'] = (torch.randn(32, 16, 3, 3) * 0.1).requires_grad_(True)
    params['conv2_b'] = torch.zeros(32, requires_grad=True)
    params['fc_w'] = (torch.randn(latent_dim, fc_in) * 0.1).requires_grad_(True)
    params['fc_b'] = torch.zeros(latent_dim, requires_grad=True)

    return params

# Step 9 - encoder_forward
import torch
import torch.nn.functional as F

def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    """
    Run a small two-layer CNN that maps pixel observations to latent embeddings.

    Args:
        obs: (B, C, H, W) float tensor of pixel observations
        encoder_params: dict with conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b

    Returns:
        (B, latent_dim) embeddings
    """
    x = F.conv2d(obs, encoder_params['conv1_w'], encoder_params['conv1_b'],
                 stride=1, padding=1)
    x = F.relu(x)

    x = F.conv2d(x, encoder_params['conv2_w'], encoder_params['conv2_b'],
                 stride=2, padding=1)
    x = F.relu(x)

    x = x.flatten(start_dim=1)

    return F.linear(x, encoder_params['fc_w'], encoder_params['fc_b'])

# Step 10 - init_target_encoder
import torch

def init_target_encoder(encoder_params: dict) -> dict:
    """
    Create the EMA target encoder as a detached deep copy of the online encoder.

    Args:
        encoder_params: dict with conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b

    Returns:
        New dict with the same keys. Each value is an independent clone
        (own storage) with requires_grad=False.
    """
    return {k: v.detach().clone() for k, v in encoder_params.items()}

# Step 11 - ema_update
import torch

def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    """
    Refresh the target encoder params as an EMA of the online encoder params.

    Args:
        target_params: dict of target tensors (not modified)
        encoder_params: dict of online tensors with the same keys (not modified)
        tau: decay in (0, 1]; higher means the target moves more slowly

    Returns:
        New dict where each value is tau * target + (1 - tau) * online,
        with no gradient tracking.
    """
    new_params = {}
    with torch.no_grad():
        for key in target_params:
            new_params[key] = tau * target_params[key] + (1.0 - tau) * encoder_params[key]
    return new_params

# Step 12 - encode_batch
import torch

def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    """
    Batch-encode observations into latent embeddings.

    Args:
        obs: (B, C, H, W) float tensor of pixel observations
        encoder_params: dict with conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b

    Returns:
        (B, latent_dim) embeddings, with gradients flowing through the params.
    """
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
import torch

def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    """
    Initialize parameters of the action-conditioned MLP predictor.

    Args:
        latent_dim: size of the latent embedding
        action_dim: number of discrete actions
        hidden_dim: hidden width of the MLP
        seed: RNG seed

    Returns:
        dict with 'action_embed_w' (action_dim, latent_dim),
        'fc1_w' (hidden_dim, 2*latent_dim), 'fc1_b' (hidden_dim,),
        'fc2_w' (latent_dim, hidden_dim), 'fc2_b' (latent_dim,).
        All are leaf tensors with requires_grad=True.
    """
    torch.manual_seed(seed)

    params = {}
    params['action_embed_w'] = (torch.randn(action_dim, latent_dim) * 0.02).requires_grad_(True)
    params['fc1_w'] = (torch.randn(hidden_dim, 2 * latent_dim) * 0.02).requires_grad_(True)
    params['fc1_b'] = torch.zeros(hidden_dim, requires_grad=True)
    params['fc2_w'] = (torch.randn(latent_dim, hidden_dim) * 0.02).requires_grad_(True)
    params['fc2_b'] = torch.zeros(latent_dim, requires_grad=True)

    return params

# Step 14 - embed_action
import torch

def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    """
    Embed discrete actions into continuous vectors via a learned table.

    Args:
        actions: (B,) long tensor of action indices in [0, action_dim)
        predictor_params: dict containing 'action_embed_w' of shape (action_dim, emb_dim)

    Returns:
        (B, emb_dim) float tensor of action embeddings.
    """
    return predictor_params['action_embed_w'][actions]

# Step 15 - predictor_forward
import torch
import torch.nn.functional as F

def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    """
    Forward pass of the action-conditioned dynamics predictor.

    Args:
        embeddings: (B, latent_dim) current latent embeddings
        actions: (B,) long tensor of discrete action indices
        predictor_params: dict with action_embed_w, fc1_w, fc1_b, fc2_w, fc2_b

    Returns:
        (B, latent_dim) predicted next embeddings.
    """
    action_emb = embed_action(actions, predictor_params)

    x = torch.cat([embeddings, action_emb], dim=-1)

    h = F.linear(x, predictor_params['fc1_w'], predictor_params['fc1_b'])
    h = torch.relu(h)

    out = F.linear(h, predictor_params['fc2_w'], predictor_params['fc2_b'])

    return out

# Step 16 - predict_next_embedding
import torch

def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    """
    Public dynamics interface: predict the next latent embedding from the
    current embedding and a discrete action, entirely in latent space.

    Args:
        embeddings: (B, latent_dim) current embeddings
        actions: (B,) long tensor of discrete action indices
        predictor_params: dict with action_embed_w, fc1_w, fc1_b, fc2_w, fc2_b

    Returns:
        (B, latent_dim) predicted next embeddings.
    """
    return predictor_forward(embeddings, actions, predictor_params)

# Step 17 - prediction_loss
import torch

def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    JEPA prediction loss: mean squared error between predicted and target embeddings.

    Args:
        predicted: (B, D) predicted next embeddings, from the online predictor
        target: (B, D) target embeddings, from the (detached) target encoder

    Returns:
        0-d scalar tensor: mean of all squared element-wise differences.
    """
    return ((predicted - target) ** 2).mean()

# Step 18 - variance_loss
import torch

def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg variance hinge loss: penalize latent dimensions whose per-batch
    std falls below gamma, to prevent representation collapse.

    Args:
        embeddings: (B, D) batch of latent embeddings
        gamma: target std floor (default 1.0)
        eps: numerical stability constant, added inside the sqrt

    Returns:
        0-d scalar tensor: mean ReLU hinge over the D dimensions.
    """
    var = embeddings.var(dim=0)          # (D,), unbiased
    std = torch.sqrt(var + eps)          # (D,)
    hinge = torch.relu(gamma - std)      # (D,)
    return hinge.mean()

# Step 19 - covariance_loss
import torch

def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """
    VICReg covariance loss: penalize off-diagonal correlations between
    latent dimensions to encourage non-redundant, decorrelated features.

    Args:
        embeddings: (B, D) batch of latent embeddings

    Returns:
        0-d scalar tensor: sum of squared off-diagonal covariance entries, divided by D.
    """
    B, D = embeddings.shape

    z = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (B - 1)

    off_diag_sq_sum = cov.pow(2).sum() - cov.diagonal().pow(2).sum()

    return off_diag_sq_sum / D

# Step 20 - vicreg_regularizer
import torch

def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    """
    Combine the VICReg variance and covariance terms into a single
    anti-collapse regularizer on a batch of embeddings.

    Args:
        embeddings: (B, D) batch of online embeddings
        var_weight: weight on the variance hinge term (default 1.0)
        cov_weight: weight on the covariance term (default 0.04)
        gamma: variance hinge threshold, forwarded to variance_loss (default 1.0)

    Returns:
        0-d scalar tensor: var_weight * variance_loss + cov_weight * covariance_loss.
    """
    var_term = variance_loss(embeddings, gamma=gamma)
    cov_term = covariance_loss(embeddings)

    return var_weight * var_term + cov_weight * cov_term

# Step 21 - jepa_loss
import torch

def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    """
    Compose the full JEPA training objective: prediction MSE plus VICReg
    regularization on the online embeddings.

    Args:
        predicted: (B, D) predictor output
        target: (B, D) stop-gradient target encoder embeddings
        online_embeddings: (B, D) online encoder output for the current observation
        pred_weight: weight on the prediction term (default 1.0)
        var_weight: weight on the variance hinge, passed to vicreg_regularizer (default 1.0)
        cov_weight: weight on the covariance term, passed to vicreg_regularizer (default 0.04)

    Returns:
        0-d scalar tensor: pred_weight * MSE(predicted, target) + VICReg(online_embeddings).
    """
    pred_loss = prediction_loss(predicted, target)
    reg = vicreg_regularizer(online_embeddings, var_weight=var_weight, cov_weight=cov_weight)

    return pred_weight * pred_loss + reg

# Step 22 - collapse_metric
import torch

def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Diagnostic metric for representation collapse: the mean of per-dimension
    standard deviations across the batch.

    Args:
        embeddings: (B, D) batch of latent embeddings

    Returns:
        0-d scalar tensor: mean over D of std(embeddings[:, d]) (unbiased, dim=0).
    """
    return embeddings.std(dim=0).mean()

# Step 23 - jepa_training_step
import torch

def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    """
    Perform one full JEPA gradient update on a batch of transitions.

    Args:
        batch: dict with 'observations' (B,C,H,W), 'actions' (B,), 'next_observations' (B,C,H,W)
        encoder_params: online encoder param dict (trainable)
        target_params: target encoder param dict (EMA teacher, not trained by SGD)
        predictor_params: predictor param dict (trainable)
        lr: SGD learning rate
        tau: EMA decay for the target encoder update

    Returns:
        (updated_encoder_params, updated_target_params, updated_predictor_params,
         loss_value, collapse_value), with the last two as Python floats.
    """
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']

    # 1. Online embeddings carry gradients.
    online = encode_batch(obs, encoder_params)

    # 2. Target embeddings are stop-gradient.
    with torch.no_grad():
        target = encode_batch(next_obs, target_params)

    # 3. Predict next embedding from current online latents + actions.
    predicted = predict_next_embedding(online, actions, predictor_params)

    # 4. JEPA objective: prediction MSE + VICReg on online embeddings.
    loss = jepa_loss(predicted, target, online)

    # Clear any stale gradients before backprop.
    for p in list(encoder_params.values()) + list(predictor_params.values()):
        p.grad = None

    loss.backward()

    def sgd_update(params: dict) -> dict:
        new_params = {}
        for k, p in params.items():
            if p.grad is not None:
                new_p = (p - lr * p.grad).detach().requires_grad_(True)
            else:
                new_p = p.detach().clone().requires_grad_(True)
            new_params[k] = new_p
        return new_params

    updated_encoder_params = sgd_update(encoder_params)
    updated_predictor_params = sgd_update(predictor_params)

    # 5. EMA-update the target encoder toward the just-updated online encoder.
    updated_target_params = ema_update(target_params, updated_encoder_params, tau=tau)

    loss_value = loss.item()
    collapse_value = collapse_metric(online.detach()).item()

    return updated_encoder_params, updated_target_params, updated_predictor_params, loss_value, collapse_value

# Step 24 - train_jepa
import torch

def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    """
    Train the JEPA world model for num_steps gradient updates.

    Args:
        dataset: dict with 'observations' (N,C,H,W), 'actions' (N,), 'next_observations' (N,C,H,W)
        encoder_params, target_params, predictor_params: initial parameter dicts
        num_steps: number of gradient updates
        batch_size: mini-batch size (sampled with replacement)
        lr: SGD learning rate
        tau: EMA decay for the target encoder
        seed: RNG seed for reproducible batch sampling

    Returns:
        (encoder_params, target_params, predictor_params, history), where history
        is a list of {'loss': float, 'collapse': float} dicts, one per step.
    """
    torch.manual_seed(seed)

    N = dataset['observations'].shape[0]
    history = []

    for _ in range(num_steps):
        idx = torch.randint(0, N, (batch_size,))

        batch = {
            'observations': dataset['observations'][idx],
            'actions': dataset['actions'][idx],
            'next_observations': dataset['next_observations'][idx],
        }

        encoder_params, target_params, predictor_params, loss_value, collapse_value = jepa_training_step(
            batch, encoder_params, target_params, predictor_params, lr=lr, tau=tau
        )

        history.append({'loss': loss_value, 'collapse': collapse_value})

    return encoder_params, target_params, predictor_params, history

# Step 25 - rollout_latent_dynamics
import torch

def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    """
    Roll out multi-step latent dynamics via the action-conditioned predictor.

    Args:
        initial_embedding: (D,) or (B, D) starting embedding
        actions: (T,) or (B, T) discrete action indices
        predictor_params: predictor parameter dict

    Returns:
        (T+1, D) if unbatched, or (T+1, B, D) if batched.
        Index 0 is the initial embedding; index t+1 is the predictor's output
        after applying actions[..., t].
    """
    was_unbatched = initial_embedding.dim() == 1

    if was_unbatched:
        current = initial_embedding.unsqueeze(0)          # (1, D)
    else:
        current = initial_embedding                       # (B, D)

    B = current.shape[0]

    if actions.dim() == 1:
        actions = actions.unsqueeze(0).expand(B, -1)       # (B, T)

    T = actions.shape[1]

    trajectory = [current]

    for t in range(T):
        current = predict_next_embedding(current, actions[:, t], predictor_params)
        trajectory.append(current)

    traj = torch.stack(trajectory, dim=0)  # (T+1, B, D)

    if was_unbatched:
        traj = traj.squeeze(1)  # (T+1, D)

    return traj

# Step 26 - multi_step_prediction_error
import torch

def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    """
    Multi-step latent dynamics accuracy: mean MSE between rolled-out latent
    predictions and target-encoded true future observations.

    Args:
        dataset: dict with 'observations' (N,C,H,W), 'actions' (N,), 'next_observations' (N,C,H,W)
        encoder_params: online encoder params
        target_params: EMA target encoder params
        predictor_params: predictor params
        horizon: number of steps to roll forward
        num_samples: number of trajectory windows to evaluate

    Returns:
        Python float: mean squared error over all sampled trajectories and time steps.
    """
    obs_all = dataset['observations']
    action_all = dataset['actions']
    next_obs_all = dataset['next_observations']

    N = obs_all.shape[0]
    n = min(num_samples, N - horizon)

    with torch.no_grad():
        # Start observation for each of the n windows: obs at index i.
        start_obs = obs_all[:n]                       # (n, C, H, W)

        # Action sequence of length `horizon` for each window: actions[i : i+horizon].
        action_seqs = torch.stack(
            [action_all[i:i + horizon] for i in range(n)], dim=0
        )  # (n, horizon)

        # True future observations for each window: next_obs[i : i+horizon].
        future_obs = torch.stack(
            [next_obs_all[i:i + horizon] for i in range(n)], dim=0
        )  # (n, horizon, C, H, W)

        # Encode starting observations with the online encoder.
        z0 = encode_batch(start_obs, encoder_params)   # (n, D)

        # Roll the predictor forward.
        pred_traj = rollout_latent_dynamics(z0, action_seqs, predictor_params)  # (horizon+1, n, D)
        pred_future = pred_traj[1:]                     # (horizon, n, D)

        # Encode true future observations with the target encoder.
        C, H, W = future_obs.shape[2], future_obs.shape[3], future_obs.shape[4]
        flat_future_obs = future_obs.reshape(n * horizon, C, H, W)
        flat_true = encode_batch(flat_future_obs, target_params)  # (n*horizon, D)
        D = flat_true.shape[-1]
        true_future = flat_true.reshape(n, horizon, D).permute(1, 0, 2)  # (horizon, n, D)

        mse = torch.mean((pred_future - true_future) ** 2)

    return mse.item()

# Step 27 - init_linear_probe
import torch

def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    """
    Initialize a linear probe mapping latent embeddings to true agent state (x, y).

    Args:
        latent_dim: size of the latent embedding
        state_dim: size of the true state (default 2, for (x, y))
        seed: RNG seed

    Returns:
        dict with 'w' of shape (state_dim, latent_dim) and 'b' of shape (state_dim,).
    """
    torch.manual_seed(seed)

    w = torch.randn(state_dim, latent_dim) * 0.01
    b = torch.zeros(state_dim)

    return {'w': w, 'b': b}

# Step 28 - train_linear_probe
import torch

def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    """
    Fit a linear probe via full-batch gradient descent on frozen embeddings.

    Args:
        embeddings: (N, latent_dim) frozen embeddings (not trained)
        states: (N, state_dim) true agent states
        probe_params: dict with 'w' (latent_dim, state_dim), 'b' (state_dim,)
        num_steps: number of gradient descent steps
        lr: learning rate

    Returns:
        dict with updated 'w' and 'b', detached leaf tensors.
    """
    embeddings = embeddings.detach()
    states = states.detach().float()

    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        pred = embeddings @ w + b
        loss = torch.mean((pred - states) ** 2)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad = None
        b.grad = None

    return {'w': w.detach().clone(), 'b': b.detach().clone()}

# Step 29 - probe_state_recovery
import torch

def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    """
    Evaluate how well JEPA latents recover true agent state via a linear probe.
    """
    observations = dataset['observations']
    states = dataset['states'].float()

    with torch.no_grad():
        embeddings = encode_batch(observations, encoder_params)

    D = embeddings.shape[-1]
    state_dim = states.shape[-1]

    if probe_params is None:
        probe_params = init_linear_probe(latent_dim=D, state_dim=state_dim, seed=0)

    # probe_params['w'] is (state_dim, latent_dim); train_linear_probe expects
    # (latent_dim, state_dim), so transpose going in and coming back out.
    transposed_in = {'w': probe_params['w'].T.contiguous(), 'b': probe_params['b']}
    trained = train_linear_probe(embeddings, states, transposed_in, num_steps=num_probe_steps)
    trained_probe = {'w': trained['w'].T.contiguous(), 'b': trained['b']}

    w = trained_probe['w']
    b = trained_probe['b']

    with torch.no_grad():
        pred = embeddings @ w.T + b
        mse = torch.mean((pred - states) ** 2)
        mean_abs_error = torch.mean(torch.abs(pred - states))

    return {
        'mse': mse.item(),
        'mean_abs_error': mean_abs_error.item(),
        'probe_params': trained_probe,
    }

# Step 30 - encode_goal
import torch

def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    """
    Convert a desired agent position into a latent goal embedding.

    Args:
        goal_state: (2,) tensor of (x, y) goal coordinates
        encoder_params: encoder parameter dict
        room_size: side length of the square room

    Returns:
        (latent_dim,) embedding of the goal position.
    """
    goal_obs = render_observation(goal_state, room_size=room_size)  # (1, H, W)
    goal_obs = goal_obs.unsqueeze(0)                                 # (1, 1, H, W)

    with torch.no_grad():
        goal_embedding = encoder_forward(goal_obs, encoder_params)  # (1, latent_dim)

    return goal_embedding.squeeze(0)  # (latent_dim,)

# Step 31 - latent_cost
import torch

def latent_cost(latents, goal_embedding):
    """
    Squared L2 distance from each latent embedding to a goal embedding.

    Args:
        latents: [..., D] tensor of embeddings
        goal_embedding: [D] (or broadcastable) goal embedding

    Returns:
        [...] tensor of squared L2 distances, feature axis reduced.
    """
    return ((latents - goal_embedding) ** 2).sum(dim=-1)

# Step 32 - sample_action_sequences
import torch

def sample_action_sequences(n_sequences, horizon, n_actions):
    """
    Sample random discrete action sequences for random-shooting MPC.

    Args:
        n_sequences: number of candidate action plans
        horizon: length of each plan
        n_actions: number of discrete actions (values drawn from [0, n_actions))

    Returns:
        (n_sequences, horizon) torch.int64 tensor of action indices.
    """
    return torch.randint(0, n_actions, (n_sequences, horizon))

# Step 33 - score_action_sequences
import torch

def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    """
    Score candidate action sequences for random-shooting MPC by cumulative
    latent distance to a goal, entirely in embedding space.

    Args:
        start_embedding: (D,) shared starting embedding
        action_sequences: (N, H) candidate discrete action plans
        goal_embedding: (D,) goal embedding
        predictor_params: predictor parameter dict

    Returns:
        (N,) tensor of total cost per candidate sequence.
    """
    N = action_sequences.shape[0]

    start_batch = start_embedding.unsqueeze(0).expand(N, -1)  # (N, D)

    with torch.no_grad():
        trajectory = rollout_latent_dynamics(start_batch, action_sequences, predictor_params)  # (H+1, N, D)
        predicted = trajectory[1:]  # (H, N, D)

        step_costs = latent_cost(predicted, goal_embedding)  # (H, N)
        total_costs = step_costs.sum(dim=0)  # (N,)

    return total_costs

# Step 34 - select_best_plan
import torch

def select_best_plan(action_sequences, costs):
    """
    Select the action sequence with the lowest cost from a batch of candidates.

    Args:
        action_sequences: (N, H) candidate action plans
        costs: (N,) cost per candidate

    Returns:
        (H,) the winning action sequence, same dtype as action_sequences.
    """
    best_idx = torch.argmin(costs)
    return action_sequences[best_idx]

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

