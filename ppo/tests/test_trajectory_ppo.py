"""
Test script for trajectory-based PPO implementation.

Tests basic functionality of action tokenization, trajectory buffer, and PPO algorithms.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add repo root so repo-local packages resolve when running the file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vla_oft.min_vla.action_tokenizer import ActionTokenizer
from ppo.trajectory_buffer import TrajectoryBuffer
from ppo.core_algos import logprobs_from_logits, compute_policy_loss, apply_mask_with_grad_control


def test_action_tokenizer():
    """Test action tokenization round-trip."""
    print("\n" + "="*70)
    print("Testing ActionTokenizer")
    print("="*70)
    
    tokenizer = ActionTokenizer(vocab_size=32000, n_bins=256)
    print(f"✓ Initialized: {tokenizer}")
    
    # Test single action
    action = np.array([0.5, -0.3, 0.8, -1.0, 1.0, 0.0, 0.2])
    tokens = tokenizer.discretize_actions(action)
    reconstructed = tokenizer.detokenize_actions(tokens)
    
    print(f"\nOriginal action:      {action}")
    print(f"Token IDs:            {tokens}")
    print(f"Reconstructed action: {reconstructed}")
    print(f"Reconstruction error: {np.abs(action - reconstructed).mean():.6f}")
    
    assert tokens.min() >= 31744 and tokens.max() < 32000, "Token IDs out of range!"
    assert np.abs(action - reconstructed).mean() < 0.01, "Reconstruction error too large!"
    
    # Test batch
    batch_actions = np.random.uniform(-1, 1, size=(10, 7))
    batch_tokens = tokenizer.discretize_actions(batch_actions)
    batch_reconstructed = tokenizer.detokenize_actions(batch_tokens)
    batch_error = np.abs(batch_actions - batch_reconstructed).mean()
    
    print(f"\nBatch reconstruction error: {batch_error:.6f}")
    assert batch_error < 0.01, "Batch reconstruction error too large!"
    
    print("✅ ActionTokenizer tests passed!")


def test_trajectory_buffer():
    """Test trajectory buffer storage and retrieval."""
    print("\n" + "="*70)
    print("Testing TrajectoryBuffer")
    print("="*70)
    
    buffer = TrajectoryBuffer()
    print("✓ Initialized TrajectoryBuffer")
    
    # Simulate 2 trajectories with different lengths
    trajectory_lengths = [50, 30]
    device = torch.device("cpu")
    
    for traj_idx, traj_len in enumerate(trajectory_lengths):
        print(f"\nSimulating trajectory {traj_idx+1} (length={traj_len})...")
        
        for step in range(traj_len):
            # Dummy data
            obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "proprio": np.zeros(8)}
            responses = torch.randint(31744, 32000, (56,))  # 7 dims * 8 chunk
            input_ids = torch.randint(0, 32000, (1, 50))
            attention_mask = torch.ones((1, 50))
            pixel_values = torch.randn((1, 3, 224, 224))
            proprio = np.zeros(8)
            action = np.random.uniform(-1, 1, 7)
            
            # Sparse reward: only at end
            reward = 1.0 if step == traj_len - 1 else 0.0
            done = (step == traj_len - 1)
            value = 0.5
            log_prob = torch.tensor(-2.0)
            
            buffer.add(
                obs=obs,
                responses=responses,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                proprio=proprio,
                action=action,
                l1_action=action,  # Warmup-like sample with valid BC target
                reward=reward,
                done=done,
                value=value,
                old_log_prob=log_prob,
            )
    
    print(f"\n✓ Added {len(buffer)} trajectories")
    
    # Test trajectory mask generation
    mask1 = buffer.generate_traj_mask(50, 49, device)
    mask2 = buffer.generate_traj_mask(30, 29, device)
    
    assert mask1.sum() == 50, f"Mask1 wrong size: {mask1.sum()}"
    assert mask2.sum() == 30, f"Mask2 wrong size: {mask2.sum()}"
    print("✓ Trajectory masks generated correctly")
    
    # Compute advantages
    buffer.compute_advantages(gamma=0.99, verifier_gamma=1.0)
    print("✓ GRPO advantages computed")
    
    # Get data
    data = buffer.get()
    
    print(f"\nBuffer statistics:")
    print(f"  - Total steps: {len(data['observations'])}")
    print(f"  - Responses shape: {data['responses'].shape}")
    print(f"  - Actions shape: {data['actions'].shape}")
    print(f"  - Advantages shape: {data['advantages'].shape}")
    print(f"  - Rewards (non-zero): {(data['rewards'] != 0).sum()}/{len(data['rewards'])}")
    
    assert len(data['observations']) == 80, "Wrong total steps!"
    assert (data['rewards'] != 0).sum() == 2, "Wrong number of non-zero rewards!"
    assert data['bc_l1_actions'] is not None, "Expected BC-aligned L1 actions!"
    assert len(data['bc_observations']) == len(data['bc_l1_actions']), "BC observations/targets misaligned!"
    
    print("✅ TrajectoryBuffer tests passed!")


def test_trajectory_buffer_mixed_l1_hardening():
    """Test buffer hardening with mixed valid/None/invalid L1 action entries."""
    print("\n" + "="*70)
    print("Testing TrajectoryBuffer L1 hardening")
    print("="*70)

    buffer = TrajectoryBuffer()

    for step in range(4):
        obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "proprio": np.zeros(8)}
        responses = torch.randint(31744, 32000, (56,))
        input_ids = torch.randint(0, 32000, (1, 40))
        attention_mask = torch.ones((1, 40))
        pixel_values = torch.randn((1, 3, 224, 224))
        proprio = np.zeros(8)
        action = np.random.uniform(-1, 1, (8, 7))

        if step == 0:
            l1_action = action.copy()  # valid
        elif step == 1:
            l1_action = None  # missing
        elif step == 2:
            l1_action = action.copy().reshape(-1)  # wrong shape but same size (reshapable)
        else:
            l1_action = np.random.uniform(-1, 1, (7,))  # invalid shape/size (dropped)

        done = (step == 3)
        reward = 1.0 if done else 0.0

        buffer.add(
            obs=obs,
            responses=responses,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            proprio=proprio,
            action=action,
            l1_action=l1_action,
            reward=reward,
            done=done,
            value=0.0,
            old_log_prob=torch.tensor(-2.0),
        )

    buffer.compute_advantages()
    data = buffer.get()

    print(f"  - Total samples: {len(data['observations'])}")
    print(f"  - Valid BC samples: {0 if data['bc_l1_actions'] is None else len(data['bc_l1_actions'])}")
    print(f"  - L1 valid mask sum: {int(data['l1_actions_mask'].sum())}")

    assert len(data['observations']) == 4
    assert data['l1_actions_mask'].shape[0] == 4
    assert data['bc_l1_actions'] is not None
    assert len(data['bc_observations']) == len(data['bc_l1_actions'])
    assert len(data['bc_l1_actions']) == int(data['l1_actions_mask'].sum())

    print("✅ TrajectoryBuffer hardening tests passed!")


def test_trajectory_buffer_partial_chunk_metadata():
    """Test explicit metadata for terminal partial chunks."""
    print("\n" + "="*70)
    print("Testing TrajectoryBuffer partial chunk metadata")
    print("="*70)

    buffer = TrajectoryBuffer()

    full_action = np.random.uniform(-1, 1, (8, 7))
    partial_action = np.random.uniform(-1, 1, (8, 7))

    common_kwargs = {
        "obs": {"image": np.zeros((224, 224, 3), dtype=np.uint8), "proprio": np.zeros(8)},
        "input_ids": torch.randint(0, 32000, (1, 32)),
        "attention_mask": torch.ones((1, 32)),
        "pixel_values": torch.randn((1, 3, 224, 224)),
        "proprio": np.zeros(8),
        "value": 0.0,
    }

    buffer.add(
        responses=torch.randint(31744, 32000, (56,)),
        action=full_action,
        l1_action=full_action.copy(),
        reward=0.0,
        done=True,
        old_log_prob=torch.tensor(-1.5),
        response_mask=torch.ones(56, dtype=torch.bool),
        action_mask=np.ones(8, dtype=bool),
        executed_action_count=8,
        executed_token_count=56,
        chunk_is_partial=False,
        **common_kwargs,
    )

    partial_response_mask = torch.zeros(56, dtype=torch.bool)
    partial_response_mask[:21] = True
    partial_action_mask = np.zeros(8, dtype=bool)
    partial_action_mask[:3] = True
    buffer.add(
        responses=torch.randint(31744, 32000, (56,)),
        action=partial_action,
        l1_action=partial_action.copy(),
        reward=1.0,
        done=True,
        old_log_prob=torch.tensor(-0.7),
        response_mask=partial_response_mask,
        action_mask=partial_action_mask,
        executed_action_count=3,
        executed_token_count=21,
        chunk_is_partial=True,
        **common_kwargs,
    )

    buffer.compute_advantages()
    data = buffer.get()

    assert data["response_masks"].shape == data["responses"].shape
    assert data["action_masks"].shape == (2, 8)
    assert np.array_equal(data["executed_action_counts"], np.array([8, 3], dtype=np.int32))
    assert np.array_equal(data["executed_token_counts"], np.array([56, 21], dtype=np.int32))
    assert np.array_equal(data["chunk_is_partial"], np.array([False, True]))
    assert int(data["response_masks"][1].sum().item()) == 21
    assert int(data["action_masks"][1].sum()) == 3
    assert data["advantages"].shape[0] == 2
    assert data["advantages"][1] == 1.0

    print("✅ Partial chunk metadata tests passed!")


def test_trajectory_buffer_bc_views_preserve_partial_masks():
    """Test BC-aligned views keep per-sample partial-chunk masks."""
    print("\n" + "="*70)
    print("Testing BC-aligned partial chunk masks")
    print("="*70)

    buffer = TrajectoryBuffer()
    action = np.random.uniform(-1, 1, (8, 7))
    response_mask = torch.zeros(56, dtype=torch.bool)
    response_mask[:28] = True
    action_mask = np.zeros(8, dtype=bool)
    action_mask[:4] = True

    buffer.add(
        obs={"image": np.zeros((224, 224, 3), dtype=np.uint8), "proprio": np.zeros(8)},
        responses=torch.randint(31744, 32000, (56,)),
        input_ids=torch.randint(0, 32000, (1, 24)),
        attention_mask=torch.ones((1, 24)),
        pixel_values=torch.randn((1, 3, 224, 224)),
        proprio=np.zeros(8),
        action=action,
        l1_action=action.copy(),
        reward=1.0,
        done=True,
        value=0.0,
        old_log_prob=torch.tensor(-1.0),
        response_mask=response_mask,
        action_mask=action_mask,
        executed_action_count=4,
        executed_token_count=28,
        chunk_is_partial=True,
    )

    buffer.compute_advantages()
    data = buffer.get()

    assert data["bc_l1_actions"] is not None
    assert data["bc_action_masks"] is not None
    assert isinstance(data["bc_response_masks"], torch.Tensor)
    assert data["bc_l1_actions"].shape == (1, 8, 7)
    assert data["bc_action_masks"].shape == (1, 8)
    assert data["bc_response_masks"].shape == (1, 56)
    assert int(data["bc_action_masks"][0].sum()) == 4
    assert int(data["bc_response_masks"][0].sum().item()) == 28

    print("✅ BC-aligned partial mask tests passed!")


def test_ppo_algorithms():
    """Test PPO algorithm functions."""
    print("\n" + "="*70)
    print("Testing PPO Algorithms")
    print("="*70)
    
    # Test logprobs_from_logits
    batch_size, seq_len, vocab_size = 4, 10, 256
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    log_probs = logprobs_from_logits(logits, token_ids)
    
    print(f"✓ logprobs_from_logits: input {logits.shape} → output {log_probs.shape}")
    assert log_probs.shape == (batch_size, seq_len), "Wrong log_probs shape!"
    assert torch.all(log_probs <= 0), "Log probs should be negative!"
    assert log_probs.requires_grad, "Log probs should inherit gradients from logits!"
    
    # Test compute_policy_loss
    old_log_probs = torch.randn(batch_size, seq_len) - 2.0  # Negative log probs (from buffer, no grad)
    new_log_probs = torch.randn(batch_size, seq_len, requires_grad=True) - 2.0  # From policy, needs grad
    advantages = torch.randn(batch_size, seq_len)  # Computed values, no grad
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    policy_loss, clipfrac, approx_kl = compute_policy_loss(
        old_log_probs, new_log_probs, advantages, mask,
        clip_ratio_high=0.28, clip_ratio_low=0.2
    )
    
    print(f"✓ compute_policy_loss:")
    print(f"  - Policy loss: {policy_loss.item():.4f}")
    print(f"  - Clip fraction: {clipfrac.item():.4f}")
    print(f"  - Approx KL: {approx_kl.item():.4f}")
    
    assert policy_loss.requires_grad, "Policy loss should have gradients!"
    assert 0 <= clipfrac <= 1, "Clip fraction out of range!"
    
    # Test apply_mask_with_grad_control
    tensor = torch.randn(batch_size, seq_len, 10, requires_grad=True)
    mask = torch.rand(batch_size, seq_len) > 0.5
    
    masked = apply_mask_with_grad_control(tensor, mask)
    loss = masked.sum()
    loss.backward()
    
    print(f"✓ apply_mask_with_grad_control: gradients preserved")
    assert tensor.grad is not None, "Gradients should flow through mask!"
    
    print("✅ PPO algorithm tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRAJECTORY-BASED PPO IMPLEMENTATION TESTS")
    print("="*70)
    
    try:
        test_action_tokenizer()
        test_trajectory_buffer()
        test_trajectory_buffer_mixed_l1_hardening()
        test_trajectory_buffer_partial_chunk_metadata()
        test_trajectory_buffer_bc_views_preserve_partial_masks()
        test_ppo_algorithms()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nImplementation is ready for integration testing.")
        print("Next steps:")
        print("  1. Test trajectory collection with real environment")
        print("  2. Test policy update with real VLA model")
        print("  3. Run small-scale training (100 steps)")
        print("  4. Monitor for memory issues and NaN losses")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TESTS FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
