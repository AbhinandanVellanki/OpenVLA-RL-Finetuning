#!/usr/bin/env python3
"""
Sanity tests for warmup -> RL handoff logic.

Covers:
- stage locking per update
- competence gate (rolling tokenized success threshold)
- max warmup safety cap
- anti-overclone guard
"""

from collections import deque


class WarmstartGateTest:
    def __init__(
        self,
        min_steps=25000,
        max_steps=32000,
        success_threshold=0.40,
        success_window=3,
        required_consecutive=2,
        overclone_threshold=0.85,
        rollout_steps=512,
    ):
        self.use_l1_warmstart = True
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.success_window = success_window
        self.required_consecutive = required_consecutive
        self.overclone_threshold = overclone_threshold
        self.rollout_steps = rollout_steps

        self.global_step = 0
        self.training_stage = "warmup" if self.use_l1_warmstart and self.min_steps > 0 else "rl"
        self.success_history = deque(maxlen=max(1, self.success_window))
        self.consecutive_hits = 0
        self.rolling_success = None

    def _should_use_l1_actions(self, stage):
        return self.use_l1_warmstart and stage == "warmup"

    def _update_gate(self, tokenized_success_rate):
        self.success_history.append(float(tokenized_success_rate))
        self.rolling_success = sum(self.success_history) / len(self.success_history)

        if self.global_step < self.min_steps:
            self.consecutive_hits = 0
            return
        if len(self.success_history) < max(1, self.success_window):
            self.consecutive_hits = 0
            return

        if self.rolling_success >= self.success_threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

    def _should_exit_warmup(self):
        if self.global_step < self.min_steps:
            return False, "min_warmup_not_met"
        if self.max_steps > 0 and self.global_step >= self.max_steps:
            return True, "max_warmup_steps"
        if self.rolling_success is not None and self.overclone_threshold <= 1.0 and self.rolling_success >= self.overclone_threshold:
            return True, "overclone_guard"
        if self.consecutive_hits >= max(1, self.required_consecutive):
            return True, "competence_gate"
        return False, "competence_pending"

    def _determine_stage(self):
        if not self.use_l1_warmstart:
            return "rl"
        if self.training_stage == "rl":
            return "rl"
        should_exit, _ = self._should_exit_warmup()
        return "rl" if should_exit else "warmup"

    def _lock_stage_for_update(self):
        return self._determine_stage()

    def test_stage_locking(self):
        print("\n[1/4] Stage locking check")
        self.global_step = max(self.min_steps - (self.rollout_steps // 2), 0)
        locked_stage = self._lock_stage_for_update()
        assert locked_stage == "warmup"
        assert self._should_use_l1_actions(locked_stage)

        # Rollout crosses min boundary, but stage remains warmup for this update.
        self.global_step += self.rollout_steps
        next_stage = self._lock_stage_for_update()
        # Still warmup because competence gate is not passed yet.
        assert next_stage == "warmup"
        print("✓ Stage lock behavior is correct")

    def test_competence_gate(self):
        print("[2/4] Competence gate check")
        self.global_step = self.min_steps
        self.training_stage = "warmup"
        self.success_history.clear()
        self.consecutive_hits = 0
        self.rolling_success = None

        # Sequence yields rolling means >= 0.40 for two consecutive validations:
        # [0.31, 0.42, 0.50] -> 0.41 (hit 1), then [0.42, 0.50, 0.45] -> 0.456... (hit 2)
        for success in [0.31, 0.42, 0.50, 0.45]:
            self._update_gate(success)

        should_exit, reason = self._should_exit_warmup()
        assert should_exit and reason == "competence_gate"
        self.training_stage = self._determine_stage()
        assert self.training_stage == "rl"
        print("✓ Competence gate triggers warmup exit")

    def test_max_warmup_cap(self):
        print("[3/4] Max warmup cap check")
        self.global_step = self.max_steps
        self.training_stage = "warmup"
        self.success_history.clear()
        self.consecutive_hits = 0
        self.rolling_success = 0.10
        should_exit, reason = self._should_exit_warmup()
        assert should_exit and reason == "max_warmup_steps"
        print("✓ Max warmup safety cap works")

    def test_overclone_guard(self):
        print("[4/4] Overclone guard check")
        self.global_step = self.min_steps
        self.training_stage = "warmup"
        self.success_history.clear()
        self.consecutive_hits = 0
        self.rolling_success = self.overclone_threshold + 0.01
        should_exit, reason = self._should_exit_warmup()
        assert should_exit and reason == "overclone_guard"
        print("✓ Overclone guard works")


if __name__ == "__main__":
    tester = WarmstartGateTest()
    tester.test_stage_locking()
    tester.test_competence_gate()
    tester.test_max_warmup_cap()
    tester.test_overclone_guard()
    print("\n✅ Warmup handoff tests passed")
