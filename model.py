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

# Step 15 - predictor_forward (not yet solved)
# TODO: implement

# Step 16 - predict_next_embedding (not yet solved)
# TODO: implement

# Step 17 - prediction_loss (not yet solved)
# TODO: implement

# Step 18 - variance_loss (not yet solved)
# TODO: implement

# Step 19 - covariance_loss (not yet solved)
# TODO: implement

# Step 20 - vicreg_regularizer (not yet solved)
# TODO: implement

# Step 21 - jepa_loss (not yet solved)
# TODO: implement

# Step 22 - collapse_metric (not yet solved)
# TODO: implement

# Step 23 - jepa_training_step (not yet solved)
# TODO: implement

# Step 24 - train_jepa (not yet solved)
# TODO: implement

# Step 25 - rollout_latent_dynamics (not yet solved)
# TODO: implement

# Step 26 - multi_step_prediction_error (not yet solved)
# TODO: implement

# Step 27 - init_linear_probe (not yet solved)
# TODO: implement

# Step 28 - train_linear_probe (not yet solved)
# TODO: implement

# Step 29 - probe_state_recovery (not yet solved)
# TODO: implement

# Step 30 - encode_goal (not yet solved)
# TODO: implement

# Step 31 - latent_cost (not yet solved)
# TODO: implement

# Step 32 - sample_action_sequences (not yet solved)
# TODO: implement

# Step 33 - score_action_sequences (not yet solved)
# TODO: implement

# Step 34 - select_best_plan (not yet solved)
# TODO: implement

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

