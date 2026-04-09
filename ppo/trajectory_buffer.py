"""
trajectory_buffer.py

Trajectory-based rollout buffer for PPO training with OpenVLA.
Stores complete episodes with variable lengths using masking.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch


class TrajectoryBuffer:
    """
    Buffer for storing trajectory-based rollouts.
    
    Stores complete episodes with padding for variable lengths.
    Uses finish_step markers to indicate episode completion.
    """
    
    def __init__(self):
        self._warned_mixed_l1_actions = False
        self._warned_invalid_l1_shape = False
        self.clear()
    
    def clear(self):
        """Clear all stored data."""
        self.trajectories: List[Dict[str, Any]] = []
        self.current_trajectory: Dict[str, List[Any]] = {
            'observations': [],
            'responses': [],  # Action token IDs
            'response_masks': [],  # Valid action-token mask (for partial chunks)
            'input_ids': [],
            'attention_mask': [],
            'pixel_values': [],
            'proprio': [],
            'actions': [],  # Continuous actions (for environment)
            'action_masks': [],  # Valid executed-action mask (for partial chunks)
            'l1_actions': [],  # L1 regression actions (for BC targets)
            'rewards': [],
            'dones': [],
            'values': [],
            'old_log_probs': [],
            'executed_action_counts': [],
            'executed_token_counts': [],
            'chunk_is_partial': [],
        }
        self.episode_step = 0

    def _pad_prompt_tensors(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pad prompt tensors to a common sequence length for stacking."""
        max_len = max(ids.shape[1] for ids in input_ids_list)
        padded_input_ids: List[torch.Tensor] = []
        padded_attention_masks: List[torch.Tensor] = []

        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)

        return padded_input_ids, padded_attention_masks

    def _stack_l1_actions(
        self,
        l1_actions_list: List[Optional[np.ndarray]],
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Robustly stack optional L1 actions.

        Guarantees no mixed None/array `np.stack` failure:
        - all None -> returns (None, all-false mask)
        - mixed -> fills missing/invalid entries with zeros, returns validity mask
        """
        if not l1_actions_list:
            return None, np.array([], dtype=bool)

        valid_mask = np.zeros(len(l1_actions_list), dtype=bool)
        normalized: List[Optional[np.ndarray]] = []
        canonical_shape: Optional[Tuple[int, ...]] = None

        for idx, action in enumerate(l1_actions_list):
            if action is None:
                normalized.append(None)
                continue

            arr = np.asarray(action, dtype=np.float32)
            if canonical_shape is None:
                canonical_shape = arr.shape
            elif arr.shape != canonical_shape:
                expected_size = int(np.prod(canonical_shape))
                if arr.size == expected_size:
                    arr = arr.reshape(canonical_shape)
                else:
                    if not self._warned_invalid_l1_shape:
                        print(
                            "⚠️  WARNING: Inconsistent L1 action shape detected in buffer; "
                            "invalid entries will be dropped for BC."
                        )
                        self._warned_invalid_l1_shape = True
                    normalized.append(None)
                    continue

            normalized.append(arr)
            valid_mask[idx] = True

        if canonical_shape is None:
            return None, valid_mask

        if np.any(~valid_mask) and not self._warned_mixed_l1_actions:
            print(
                "⚠️  WARNING: Mixed missing/valid L1 actions detected; "
                "missing entries are masked out for BC."
            )
            self._warned_mixed_l1_actions = True

        fill_value = np.zeros(canonical_shape, dtype=np.float32)
        dense = [arr if arr is not None else fill_value for arr in normalized]
        return np.stack(dense), valid_mask

    def _finalize_current_trajectory(self) -> None:
        """Finalize and store the in-progress trajectory."""
        input_ids_list = self.current_trajectory['input_ids']
        attention_mask_list = self.current_trajectory['attention_mask']
        padded_input_ids, padded_attention_masks = self._pad_prompt_tensors(
            input_ids_list=input_ids_list,
            attention_mask_list=attention_mask_list,
        )

        l1_actions, l1_actions_mask = self._stack_l1_actions(self.current_trajectory['l1_actions'])

        trajectory = {
            'observations': self.current_trajectory['observations'].copy(),
            'responses': torch.stack(self.current_trajectory['responses']).detach(),
            'response_masks': torch.stack(self.current_trajectory['response_masks']).detach(),
            'input_ids': torch.stack(padded_input_ids).detach(),
            'attention_mask': torch.stack(padded_attention_masks).detach(),
            'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
            'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
            'actions': np.stack(self.current_trajectory['actions']),
            'action_masks': np.stack(self.current_trajectory['action_masks']),
            'l1_actions': l1_actions,
            'l1_actions_mask': l1_actions_mask,
            'rewards': np.array(self.current_trajectory['rewards']),
            'dones': np.array(self.current_trajectory['dones']),
            'values': np.array(self.current_trajectory['values']),
            'old_log_probs': torch.stack(self.current_trajectory['old_log_probs']).detach(),
            'executed_action_counts': np.array(self.current_trajectory['executed_action_counts'], dtype=np.int32),
            'executed_token_counts': np.array(self.current_trajectory['executed_token_counts'], dtype=np.int32),
            'chunk_is_partial': np.array(self.current_trajectory['chunk_is_partial'], dtype=bool),
            'finish_step': self.episode_step - 1,
            'traj_len': self.episode_step,
        }
        self.trajectories.append(trajectory)

        self.current_trajectory = {k: [] for k in self.current_trajectory.keys()}
        self.episode_step = 0
    
    def add(
        self,
        obs: Dict[str, Any],
        responses: torch.Tensor,  # Action token IDs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        proprio: Optional[np.ndarray],
        action: np.ndarray,  # Continuous action
        l1_action: Optional[np.ndarray],  # L1 regression action (for BC)
        reward: float,
        done: bool,
        value: float,
        old_log_prob: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[np.ndarray] = None,
        executed_action_count: Optional[int] = None,
        executed_token_count: Optional[int] = None,
        chunk_is_partial: Optional[bool] = None,
    ):
        """
        Add a single timestep to the current trajectory.
        
        When done=True, finalizes the trajectory and starts a new one.
        """
        responses = responses.detach()
        if response_mask is None:
            response_mask = torch.ones_like(responses, dtype=torch.bool)
        else:
            response_mask = response_mask.detach().to(dtype=torch.bool, device=responses.device)

        action_arr = np.asarray(action)
        if action_mask is None:
            action_entries = action_arr.shape[0] if action_arr.ndim >= 2 else 1
            action_mask = np.ones(action_entries, dtype=bool)
        else:
            action_mask = np.asarray(action_mask, dtype=bool)

        if executed_action_count is None:
            executed_action_count = int(action_mask.sum())
        if executed_token_count is None:
            executed_token_count = int(response_mask.sum().item())
        if chunk_is_partial is None:
            chunk_is_partial = bool(executed_action_count < action_mask.size)

        self.current_trajectory['observations'].append(obs)
        self.current_trajectory['responses'].append(responses)
        self.current_trajectory['response_masks'].append(response_mask.cpu())
        self.current_trajectory['input_ids'].append(input_ids)
        self.current_trajectory['attention_mask'].append(attention_mask)
        self.current_trajectory['pixel_values'].append(pixel_values)
        self.current_trajectory['proprio'].append(proprio)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['action_masks'].append(action_mask)
        self.current_trajectory['l1_actions'].append(l1_action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['values'].append(value)
        self.current_trajectory['old_log_probs'].append(old_log_prob)
        self.current_trajectory['executed_action_counts'].append(int(executed_action_count))
        self.current_trajectory['executed_token_counts'].append(int(executed_token_count))
        self.current_trajectory['chunk_is_partial'].append(bool(chunk_is_partial))
        
        self.episode_step += 1
        
        if done:
            self._finalize_current_trajectory()
    
    def finalize_partial_trajectory(self):
        """
        Finalize current partial trajectory if it exists.
        
        Called at end of rollout collection if trajectory is incomplete.
        """
        if self.episode_step > 0:
            self._finalize_current_trajectory()
    
    def generate_traj_mask(self, traj_len: int, finish_step: int, device: torch.device) -> torch.Tensor:
        """
        Generate trajectory mask from finish_step.
        
        Args:
            traj_len: Length of trajectory (may include padding)
            finish_step: Index of final valid step
            device: Device for tensor
        
        Returns:
            mask: Boolean mask, shape (traj_len,), True for valid steps
        """
        mask = torch.zeros(traj_len, dtype=torch.bool, device=device)
        mask[:finish_step + 1] = True
        return mask
    
    def compute_advantages(
        self,
        gamma: float = 0.99,
        verifier_gamma: float = 1.0,
    ):
        """
        Compute GRPO advantages for all trajectories.
        
        Uses sparse rewards at finish_step only.
        GRPO: A[i] = sum(gamma^t * r[t] for t in [i, finish_step])
        
        Args:
            gamma: Discount factor (unused in GRPO with sparse rewards)
            verifier_gamma: Verifier discount (1.0 for no discounting)
        """
        for traj in self.trajectories:
            traj_len = traj['traj_len']
            finish_step = traj['finish_step']
            rewards = traj['rewards']
            
            # Compute returns (reward-to-go from each step)
            returns = np.zeros(traj_len, dtype=np.float32)
            
            # Only reward at finish_step is non-zero (sparse rewards)
            # Propagate backward with gamma
            returns[finish_step] = rewards[finish_step]
            for t in range(finish_step - 1, -1, -1):
                returns[t] = rewards[t] + verifier_gamma * returns[t + 1]
            
            # GRPO: advantages = returns (no value baseline)
            # For sparse binary rewards (0 or 1), use ABSOLUTE advantages
            # This avoids negative advantages that cause policy instability
            advantages = returns.copy()
            
            traj['returns'] = returns
            traj['advantages'] = advantages
        
        # Collect all advantages for statistics (but DON'T normalize for sparse rewards)
        all_advantages = np.concatenate([t['advantages'] for t in self.trajectories])
        
        # Check for NaN or inf in advantages
        if np.any(np.isnan(all_advantages)) or np.any(np.isinf(all_advantages)):
            print(f"⚠️  WARNING: Found NaN or inf in advantages before normalization!")
            print(f"   NaN count: {np.isnan(all_advantages).sum()}")
            print(f"   Inf count: {np.isinf(all_advantages).sum()}")
            # Replace NaN/inf with zeros
            all_advantages = np.nan_to_num(all_advantages, nan=0.0, posinf=0.0, neginf=0.0)
        
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std()
        
        # Debug: Print advantage statistics
        print(f"\n📊 Advantage Statistics (ABSOLUTE - No Normalization):")
        print(f"   Mean: {adv_mean:.6f}")
        print(f"   Std: {adv_std:.6f}")
        print(f"   Min: {all_advantages.min():.6f}")
        print(f"   Max: {all_advantages.max():.6f}")
        print(f"   Total samples: {len(all_advantages)}")
        
        # CRITICAL: For sparse binary rewards (0 or 1), DO NOT normalize!
        # Normalization creates negative advantages which confuse the policy.
        # Instead, use absolute advantages:
        #   - Successful trajectories: advantage = 1.0 (increase log prob)
        #   - Failed trajectories: advantage = 0.0 (no gradient)
        print(f"\n✓ Using ABSOLUTE advantages (no normalization) for sparse rewards")
        print(f"  - Successful steps: advantage ≈ {all_advantages.max():.2f}")
        print(f"  - Failed steps: advantage ≈ {all_advantages.min():.2f}")
        print(f"  - This ensures policy only increases prob of successful actions\n")
        
        # Final safety check
        for traj in self.trajectories:
            if np.any(np.isnan(traj['advantages'])) or np.any(np.isinf(traj['advantages'])):
                print(f"⚠️  ERROR: NaN/inf in advantages! Setting to zeros.")
                traj['advantages'] = np.nan_to_num(traj['advantages'], nan=0.0, posinf=0.0, neginf=0.0)
    
    def get(self) -> Dict[str, Any]:
        """
        Get all stored trajectories as a dictionary.
        
        Returns:
            data: Dictionary containing all trajectory data
        """
        if not self.trajectories:
            return {
                'observations': [],
                'responses': [],
                'response_masks': [],
                'input_ids': [],
                'attention_mask': [],
                'pixel_values': [],
                'proprio': [],
                'actions': [],
                'action_masks': [],
                'l1_actions': None,
                'l1_actions_mask': np.array([], dtype=bool),
                'bc_observations': [],
                'bc_l1_actions': None,
                'bc_action_masks': None,
                'bc_response_masks': [],
                'rewards': [],
                'returns': [],
                'advantages': [],
                'old_log_probs': [],
                'action_masks': np.array([], dtype=bool),
                'response_masks': [],
                'executed_action_counts': np.array([], dtype=np.int32),
                'executed_token_counts': np.array([], dtype=np.int32),
                'chunk_is_partial': np.array([], dtype=bool),
                'finish_steps': [],
                'traj_lens': [],
            }

        proprio_chunks = [traj['proprio'] for traj in self.trajectories if traj['proprio'] is not None]
        l1_action_chunks = [traj['l1_actions'] for traj in self.trajectories if traj['l1_actions'] is not None]

        mask_chunks: List[np.ndarray] = []
        for traj in self.trajectories:
            traj_mask = traj.get('l1_actions_mask')
            if traj_mask is None:
                if traj['l1_actions'] is None:
                    traj_mask = np.zeros(traj['traj_len'], dtype=bool)
                else:
                    traj_mask = np.ones(traj['traj_len'], dtype=bool)
            mask_chunks.append(traj_mask.astype(bool))
        l1_actions_mask = np.concatenate(mask_chunks) if mask_chunks else np.array([], dtype=bool)

        # BC-aligned views (only valid L1-targeted samples).
        bc_observations: List[Dict[str, Any]] = []
        bc_l1_actions_list: List[np.ndarray] = []
        bc_action_masks_list: List[np.ndarray] = []
        bc_response_masks_list: List[torch.Tensor] = []
        for traj in self.trajectories:
            traj_l1_actions = traj['l1_actions']
            if traj_l1_actions is None:
                continue
            traj_mask = traj.get('l1_actions_mask')
            if traj_mask is None:
                traj_mask = np.ones(traj_l1_actions.shape[0], dtype=bool)
            for obs, action, action_mask, response_mask, is_valid in zip(
                traj['observations'],
                traj_l1_actions,
                traj['action_masks'],
                traj['response_masks'],
                traj_mask,
            ):
                if is_valid:
                    bc_observations.append(obs)
                    bc_l1_actions_list.append(action)
                    bc_action_masks_list.append(np.asarray(action_mask, dtype=bool))
                    bc_response_masks_list.append(response_mask.detach().clone())

        l1_actions_concat: Optional[np.ndarray] = None
        if l1_action_chunks:
            try:
                l1_actions_concat = np.concatenate(l1_action_chunks)
            except ValueError:
                if not self._warned_invalid_l1_shape:
                    print(
                        "⚠️  WARNING: Inconsistent L1 action shapes across trajectories; "
                        "global l1_actions concat disabled for this buffer snapshot."
                    )
                    self._warned_invalid_l1_shape = True
                l1_actions_concat = None

        bc_l1_actions: Optional[np.ndarray] = None
        if bc_l1_actions_list:
            canonical_shape = bc_l1_actions_list[0].shape
            filtered_obs: List[Dict[str, Any]] = []
            filtered_actions: List[np.ndarray] = []
            filtered_action_masks: List[np.ndarray] = []
            filtered_response_masks: List[torch.Tensor] = []
            for obs, action, action_mask, response_mask in zip(
                bc_observations,
                bc_l1_actions_list,
                bc_action_masks_list,
                bc_response_masks_list,
            ):
                arr = np.asarray(action, dtype=np.float32)
                if arr.shape != canonical_shape:
                    expected_size = int(np.prod(canonical_shape))
                    if arr.size == expected_size:
                        arr = arr.reshape(canonical_shape)
                    else:
                        continue
                filtered_obs.append(obs)
                filtered_actions.append(arr)
                filtered_action_masks.append(np.asarray(action_mask, dtype=bool))
                filtered_response_masks.append(response_mask)
            bc_observations = filtered_obs
            bc_l1_actions = np.stack(filtered_actions) if filtered_actions else None
            bc_action_masks = np.stack(filtered_action_masks) if filtered_action_masks else None
            bc_response_masks = torch.stack(filtered_response_masks) if filtered_response_masks else []
        else:
            bc_action_masks = None
            bc_response_masks = []

        data = {
            'observations': [obs for traj in self.trajectories for obs in traj['observations']],
            'responses': torch.cat([traj['responses'] for traj in self.trajectories]),
            'response_masks': torch.cat([traj['response_masks'] for traj in self.trajectories]),
            'input_ids': torch.cat([traj['input_ids'] for traj in self.trajectories]),
            'attention_mask': torch.cat([traj['attention_mask'] for traj in self.trajectories]),
            'pixel_values': torch.cat([traj['pixel_values'] for traj in self.trajectories]),
            'proprio': np.concatenate(proprio_chunks) if proprio_chunks else None,
            'actions': np.concatenate([traj['actions'] for traj in self.trajectories]),
            'action_masks': np.concatenate([traj['action_masks'] for traj in self.trajectories]),
            'l1_actions': l1_actions_concat,
            'l1_actions_mask': l1_actions_mask,
            'bc_observations': bc_observations,
            'bc_l1_actions': bc_l1_actions,
            'bc_action_masks': bc_action_masks,
            'bc_response_masks': bc_response_masks,
            'rewards': np.concatenate([traj['rewards'] for traj in self.trajectories]),
            'returns': np.concatenate([traj['returns'] for traj in self.trajectories]),
            'advantages': np.concatenate([traj['advantages'] for traj in self.trajectories]),
            'old_log_probs': torch.cat([traj['old_log_probs'] for traj in self.trajectories]),
            'executed_action_counts': np.concatenate([traj['executed_action_counts'] for traj in self.trajectories]),
            'executed_token_counts': np.concatenate([traj['executed_token_counts'] for traj in self.trajectories]),
            'chunk_is_partial': np.concatenate([traj['chunk_is_partial'] for traj in self.trajectories]),
            'finish_steps': [traj['finish_step'] for traj in self.trajectories],
            'traj_lens': [traj['traj_len'] for traj in self.trajectories],
        }
        
        return data
    
    def __len__(self) -> int:
        """Return total number of complete trajectories."""
        return len(self.trajectories)
