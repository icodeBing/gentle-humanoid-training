"""Threshold following, mixed loads and deployable observation contract on CPU.

These tests verify task mechanics, not that a trained policy can walk.
"""
import math
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from test_standing_following import following_command
from test_standing_stable import ForceHelpers


def threshold_command(force=25., safe=15., **schedule):
    c = following_command()
    c.threshold_following = True
    c.hand_reference_limit = .25
    c.moving_speed_threshold = .01
    c.safe_force_limit[:] = safe
    c.set_eval_stable_force(force, [1., 0., 0.])
    c.set_eval_force_schedule('stable', **schedule)
    c.root_height_ref[:] = c.asset.data.root_pos_w[:, 2:3]
    return c


class ThresholdFollowingTests(unittest.TestCase):
    def test_low_equal_and_high_force_have_expected_steady_velocity(self):
        for force, target in [(0., 0.), (10., 0.), (15., 0.), (25., 10/45), (35., 20/45)]:
            with self.subTest(force=force):
                c = threshold_command(force)
                for _ in range(500):
                    c.before_update()
                    c.step(0)
                self.assertAlmostEqual(c.root_vel_ref_w[0, 0].item(), target, places=5)
                self.assertAlmostEqual(c.force_applied_w[0, 5, 0].item(), force, places=5)
                self.assertLessEqual(c.force_drive_b.norm(dim=-1).max().item(), 15.00001)
                self.assertLessEqual(c.admit.x.norm(dim=-1).max().item(), .250001)
                if force == 10:
                    # A stationary body must still yield its hand.
                    self.assertGreater(c.force_keypoint_b[0, 5, 0].item(), .01)
                    self.assertFalse(c._moving().item())

    def test_body_command_uses_actual_spring_state_not_virtual_remaining_force(self):
        c = threshold_command()
        c.set_eval_force_schedule('spring')
        c.force_origin_b[0, 5, 0] = .25
        c.force_kp_nominal[:] = 100.
        for _ in range(350):
            c.before_update()
        self.assertAlmostEqual(c.follow_input_force_w[0, 0].item(), 25., places=4)
        self.assertAlmostEqual(c.force_expected_w[0, 5, 0].item(), 15., places=3)
        self.assertAlmostEqual(c.root_vel_ref_w[0, 0].item(), 10/45, places=4)
        c.asset.data.body_pos_w[0, 5, 0] = .15
        c.before_update()
        self.assertAlmostEqual(c.follow_input_force_w[0, 0].item(), 10., places=4)
        self.assertEqual(c.follow_excess_force_w.norm().item(), 0.)

    def test_release_brakes_and_world_force_does_not_rotate(self):
        c = threshold_command()
        for _ in range(200):
            c.before_update()
        before = c.root_vel_ref_w[0, 0].item()
        c.asset.data.root_quat_w[:] = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        c.before_update()
        c.step(0)
        torch.testing.assert_close(c.force_applied_w[0, 5], torch.tensor([25., 0., 0.]))
        torch.testing.assert_close(c.follow_excess_force_w[0], torch.tensor([10., 0., 0.]))
        c.set_eval_stable_force(0., [1., 0., 0.])
        c.before_update()
        self.assertLess(c.root_vel_ref_w[0, 0].item(), before)
        for _ in range(300):
            c.before_update()
        self.assertLess(c.root_vel_ref_w.norm().item(), .0001)
        self.assertLess(c.admit.x.norm(dim=-1).max().item(), .001)

    def test_multi_contact_threshold_is_on_net_horizontal_force(self):
        c = threshold_command()
        force = torch.zeros(1, 6, 3)
        force[0, 4:, 0] = 10.
        for _ in range(400):
            c._update_body_reference(force)
        self.assertAlmostEqual(c.root_vel_ref_w[0, 0].item(), 5/45, places=4)
        force[0, 4, 0] = -10.
        force[:, :, 2] = 100.
        c.root_vel_ref_w.zero_()
        c._update_body_reference(force)
        self.assertEqual(c.root_vel_ref_w.norm().item(), 0.)

    def test_constant_force_rewards_are_masked_but_arm_tracking_is_not(self):
        c = threshold_command()
        c.before_update()
        c.step(0)
        self.assertFalse(c.force_reward()[1].item())
        self.assertFalse(c.force_exd_penalty()[1].item())
        score = c.force_target_tracking().item()
        self.assertLess(score, 1.)
        c.asset.data.body_pos_w[:, :6] = c._reward_reference['hand_position']
        self.assertAlmostEqual(c.force_target_tracking().item(), 1.)
        c.set_eval_force_schedule('spring')
        c.before_update()
        c.step(0)
        self.assertTrue(c.force_reward()[1].item())

    def test_dense_cost_penalizes_stall_reverse_and_overspeed_and_balance_matters(self):
        c = threshold_command()
        c.root_vel_ref_w[0, 0] = .222
        costs = []
        for speed in [-.222, 0., .111, .222, .444]:
            c.asset.data.root_lin_vel_w[0, 0] = speed
            costs.append(c.body_velocity_error().item())
        self.assertLess(costs[0], costs[1])
        self.assertLess(costs[1], costs[2])
        self.assertLess(costs[2], costs[3])
        self.assertEqual(costs[3], 0.)
        self.assertLess(costs[4], 0.)
        c.asset.data.root_lin_vel_w[0, 0] = .222
        balanced = c.body_yield_velocity()[0].item()
        c.asset.data.root_pos_w[0, 2] = -.3
        self.assertLess(c.body_yield_velocity()[0].item(), balanced / 100)

    def test_rewards_use_executed_reference_and_load_type_at_event_boundary(self):
        c = threshold_command()
        c.root_vel_ref_w[0, 0] = .22
        c.asset.data.root_lin_vel_w[0, 0] = .22
        c.before_update()
        c.root_vel_ref_w[0, 0] = .22
        c.step(0)
        # Preparing a different next event must not change this step's reward.
        c.set_eval_force_schedule('spring')
        c.root_vel_ref_w[0, 0] = -.22
        c.force_keypoint_w += 1.
        self.assertAlmostEqual(c.body_velocity_error().item(), 0.)
        self.assertFalse(c.force_reward()[1].item())
        c._reward_reference_valid[:] = False  # reset must not reuse previous episode
        self.assertLess(c.body_velocity_error().item(), -1.)

    def test_student_input_does_not_contain_force_derived_references(self):
        c = threshold_command()
        baseline = c.command().clone()
        c.force_keypoint_b.uniform_(-10, 10)
        c.root_vel_ref_w.uniform_(-1, 1)
        c.force_applied_w.uniform_(-30, 30)
        torch.testing.assert_close(c.command(), baseline)
        self.assertEqual(tuple(baseline.shape), (1, 1))
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'cfg/task/G1/G1_settle.yaml').read_text())
        self.assertNotIn('yield_command', config['observation']['policy'])
        self.assertIn('yield_command', config['observation']['priv'])
        self.assertIn('force_priv', config['observation']['priv'])
        self.assertEqual(config['command']['safe_force_range'], [5., 15.])
        self.assertIn('heading_command', config['observation']['policy'])
        self.assertEqual(c.force_priv().shape[-1], 4 * 6 * 3 + 9)
        # Relative heading is state-derived and does not reveal a force target.
        heading = c.heading_command().clone()
        c.force_keypoint_b.uniform_(-100, 100)
        torch.testing.assert_close(c.heading_command(), heading)

    def test_constant_sampling_covers_below_near_above_threshold(self):
        c = threshold_command()
        torch.manual_seed(42)
        safe = torch.full((12000, 1), 15.)
        force = c._sample_constant_magnitude(safe)
        fractions = [(force < 13.5).float().mean(),
                     ((force >= 13.5) & (force <= 16.5)).float().mean(),
                     (force >= 17).float().mean()]
        for frac in fractions:
            self.assertGreater(frac.item(), .30)
            self.assertLess(frac.item(), .37)
        self.assertGreaterEqual(force.min().item(), 3.)
        self.assertLessEqual(force.max().item(), 35.)
        c.step_schedule(0.)
        self.assertAlmostEqual(c.load_curriculum_scale, .35)
        c.step_schedule(.25)
        self.assertEqual(c.load_curriculum_scale, 1.)
        c.student_train = True
        c.step_schedule(0.)
        self.assertEqual(c.load_curriculum_scale, 1.)

    def test_mixed_training_ramps_holds_releases_and_never_resurrects_idle_links(self):
        c = threshold_command()
        c.set_eval_force_schedule('random')
        c.load_is_constant[:] = True
        c.constant_force_w[0, 5, 0] = 25.
        c.force_enable[0, 5] = True
        c.load_ramp_steps[:] = 10
        c.load_hold_steps[:] = 20
        for age, expected in [(0, 2.5), (9, 25.), (19, 25.), (29, 25.), (34, 12.5), (38, 2.5)]:
            c.load_age[:] = age
            c._update_mixed_schedule()
            c.step(0)
            self.assertAlmostEqual(c.force_applied_w[0, 5, 0].item(), expected, places=4)
        c.force_enable.zero_()
        c.load_age[:] = 15
        c._update_mixed_schedule()
        c.step(0)
        self.assertEqual(c.force_applied_w.norm().item(), 0.)

    def test_sampling_new_events_uses_both_load_types_and_fixed_world_vectors(self):
        c = threshold_command()
        c.set_eval_force_schedule('random')
        c.eval_safe_force = None
        c.eval_direction_b = None
        c.hand_stiffness_range = (5., 250.)
        c.safe_force_range = (5., 15.)
        c.force_time_range = (10, 20)
        c.ramp_time_range = (3, 5)
        c.force_timer = torch.zeros(1, dtype=torch.int32)
        c.force_body_sampling = 'balanced_hands'
        c.hand_offset_range = (.1, .25)
        c.eval_force_body_mask[:] = True
        modes = set()
        for _ in range(120):
            c._sample_force(torch.tensor([0]))
            modes.add(c.load_is_constant.item())
            self.assertTrue(5 <= c.safe_force_limit.item() <= 15)
            self.assertTrue(.1 <= c.force_offset.item() <= .250001)
            self.assertEqual(c.load_age.item(), 0)
            self.assertEqual(c.force_scale.item(), 0.)
        self.assertEqual(modes, {False, True})

    def test_spring_net_cap_is_physical_without_torso_counter_force(self):
        c = threshold_command()
        c.set_eval_force_schedule('spring')
        c.eval_force_body_mask[:] = True
        c.force_origin_b[:, :, 0] = 1.
        c.before_update()
        c.step(0)
        self.assertAlmostEqual(c.force_applied_w.sum(1).norm().item(), 60., places=4)
        self.assertEqual(c.force_apply_buffer[0, 6].norm().item(), 0.)

    def test_parallel_environments_can_mix_load_types_and_thresholds(self):
        c = threshold_command()
        n = 4
        for name, value in list(vars(c).items()):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1:
                setattr(c, name, value.expand(n, *value.shape[1:]).clone())
        for name, value in vars(c.asset.data).items():
            if isinstance(value, torch.Tensor):
                setattr(c.asset.data, name, value.expand(n, *value.shape[1:]).clone())
        c.num_envs = n
        c.admit = type(c.admit)(n, 6, .005, 1, 'cpu', mass=.1, damping=2., vel_clip=4., acc_clip=1000.)
        c.set_eval_force_schedule('random')
        c.load_is_constant[:] = torch.tensor([True, False, True, False])
        c.constant_force_w[:, 5, 0] = 25.
        c.force_origin_b[:, 5, 0] = .25
        c.force_enable[:, 5] = True
        c.safe_force_limit[:, 0] = torch.tensor([5., 10., 15., 15.])
        c.load_ramp_steps[:] = 10
        c.load_hold_steps[:] = 20
        c.load_age[:] = 15
        c.before_update()
        c.step(0)
        torch.testing.assert_close(c.force_applied_w[:, 5, 0], torch.full((n,), 25.))
        torch.testing.assert_close(c.follow_excess_force_w[:, 0], torch.tensor([20., 15., 10., 10.]))
        torch.testing.assert_close(c.root_vel_ref_w[:, 0], torch.tensor([20., 15., 10., 10.]) * .02 / 35)
        self.assertEqual(c.command().shape, (n, 1))
        self.assertEqual(c.force_priv().shape, (n, 81))

    def test_airtime_reward_preserves_one_independent_score_per_environment(self):
        for n in [4, 4096]:
            with self.subTest(num_envs=n):
                c = threshold_command()
                for name, value in list(vars(c).items()):
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1:
                        setattr(c, name, value.expand(n, *value.shape[1:]).clone())
                for name, value in vars(c.asset.data).items():
                    if isinstance(value, torch.Tensor):
                        setattr(c.asset.data, name, value.expand(n, *value.shape[1:]).clone())
                c.num_envs = n
                c.root_vel_ref_w[:, 0] = .2
                c.asset.data.root_lin_vel_w[:, 0] = torch.tensor([0., .1, .2, .4]).repeat(n // 4)
                c.asset.data.root_pos_w[3::4, 2] = -.1
                c.contact_sensor.data.last_air_time = torch.full((n, 2), .3)
                c.contact_sensor.compute_first_contact = lambda dt: torch.tensor([[True, False]]).expand(n, 2)
                score, active = c.yield_step_air_time()
                self.assertEqual(score.shape, (n, 1))
                self.assertEqual(active.shape, (n, 1))
                expected = torch.tensor([0., .09, .18, .18 * math.exp(-1)]).repeat(n // 4).unsqueeze(-1)
                torch.testing.assert_close(score, expected)
                self.assertTrue(active.all())

    def test_replay_records_held_excess_and_arm_state_errors(self):
        c = threshold_command()
        c.before_update()
        held = ForceHelpers._following_reference_snapshot(c)
        force_ref = ForceHelpers._force_reference_snapshot(c.env)
        c.step(0)
        c.before_update()
        # Snapshot must survive the next command update.
        c.follow_excess_force_w.zero_()
        row = ForceHelpers._following_motion_fields(c, 0, torch.tensor([10/45, 0., 0.]), held)
        self.assertEqual(row['follow_excess_x_N'], 10.)
        self.assertEqual(row['safe_force_N'], 15.)
        self.assertEqual(row['constant_load'], 1)
        snap = ForceHelpers._force_snapshot(c.env, reference=force_ref)
        force_row = ForceHelpers._force_record_rows(snap, 0, .02, .005)[5]
        self.assertIn('hand_position_error_m', force_row)
        self.assertIn('f_drive_ref_x_N', force_row)
        self.assertAlmostEqual(force_row['f_applied_x_N'], 25.)
        expected_error = np.linalg.norm(snap['hand_pos_actual'][5] - force_ref['hand_pos_ref'][5])
        self.assertAlmostEqual(force_row['hand_position_error_m'], expected_error)


if __name__ == '__main__':
    unittest.main()
