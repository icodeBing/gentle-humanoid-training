"""Behavioral tensor checks; does not claim a trained walking policy works."""

import math
from types import SimpleNamespace
import unittest

import torch

from test_standing_stable import ForceHelpers, make_command


def following_command():
    c = make_command()
    c.following_rewards = True
    c.foot_asset_ids = [0, 1]
    c.foot_sensor_ids = [0, 1]
    c.contact_sensor = SimpleNamespace(data=SimpleNamespace(
        current_contact_time=torch.ones(1, 2), last_air_time=torch.zeros(1, 2)))
    c.contact_sensor.compute_first_contact = lambda dt: torch.zeros(1, 2, dtype=torch.bool)
    return c


class FollowingTests(unittest.TestCase):
    def test_tracking_requires_motion_in_correct_direction_at_correct_speed(self):
        c = following_command()
        c.root_vel_ref_w[0, 0] = 10 / 45
        scores = []
        for speed in [0., -10/45, 10/45, 20/45]:
            c.asset.data.root_lin_vel_w[0, 0] = speed
            score, active = c.body_yield_velocity()
            stall, _ = c.body_yield_stall()
            scores.append((score.item(), stall.item()))
            self.assertTrue(active.item())
        self.assertLess(scores[0][0], .01)
        self.assertEqual(scores[0][1], -1.)
        self.assertEqual(scores[1][1], -2.)
        self.assertAlmostEqual(scores[2][0], 1.)
        self.assertEqual(scores[2][1], 0.)
        self.assertLess(scores[3][0], .01)
        c.asset.data.root_lin_vel_w[:] = torch.tensor([[0., 10/45, 0.]])
        self.assertLess(c.body_yield_velocity()[0].item(), .01)

    def test_braking_reference_stays_a_moving_target_after_force_release(self):
        c = following_command()
        c.force_expected_w.zero_()
        c.root_vel_ref_w[0, 0] = .15
        self.assertTrue(c.body_yield_velocity()[1].item())
        self.assertFalse(c.body_rest_velocity()[1].item())
        c.root_vel_ref_w.zero_()
        self.assertFalse(c.body_yield_velocity()[1].item())
        self.assertTrue(c.body_rest_velocity()[1].item())

    def test_only_enabled_points_determine_force_reward(self):
        c = following_command()
        c.force_enable.zero_()
        c.force_enable[0, 5] = True
        c.force_applied_w[0, 5, 0] = 8.
        c.force_applied_w[0, 0, 0] = 100.  # inactive buffer must not dilute or dominate
        score, valid = c.force_reward()
        self.assertAlmostEqual(score.item(), math.exp(-2), places=6)
        self.assertTrue(valid.item())
        c.force_enable.zero_()
        self.assertFalse(c.force_reward()[1].item())
        self.assertFalse(c.force_exd_penalty()[1].item())

    def test_yaw_turn_has_a_cost_and_command_exposes_heading_error(self):
        c = following_command()
        c.asset.data.root_quat_w[:] = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        self.assertLess(c.body_heading_tracking().item(), 1e-8)
        command = c.yield_command()
        self.assertAlmostEqual(command[0, -1].item(), -1., places=5)
        c.asset.data.root_ang_vel_w[0, 2] = 1.4
        self.assertAlmostEqual(c.body_yaw_rate().item(), -1.96, places=5)

    def test_offset_mixture_varies_across_events_but_fixed_override_wins(self):
        c = following_command()
        c.hand_offset_range = (.1, .25)
        torch.manual_seed(31)
        sample = c._sample_offset(1000).flatten()
        self.assertTrue(((sample >= .1) & (sample <= .25)).all())
        nominal_count = (sample == .2).sum().item()
        self.assertTrue(400 < nominal_count < 600)
        c.eval_spring_offset = .15
        torch.testing.assert_close(c._sample_offset(50), torch.full((50, 1, 1), .15))

    def test_offset_event_is_preserved_and_shared_by_all_points(self):
        c = following_command()
        c.hand_offset_range = (.1, .25)
        c.hand_offset_random_prob = 1.
        c.hand_stiffness_range = (5., 250.)
        c.safe_force_range = (5., 15.)
        c.force_time_range = (300, 900)
        c.force_timer = torch.zeros(1, dtype=torch.int32)
        c.eval_direction_b = None
        c._sample_force(torch.tensor([0]))
        sampled = c.force_offset.clone()
        distances = (c.force_origin_b-c.nominal_hand_b).norm(dim=-1)
        torch.testing.assert_close(distances, sampled.reshape(1, 1).expand(1, 6))
        c.nominal_initialized.zero_()
        c.before_update()
        torch.testing.assert_close(c.force_offset, sampled)
        torch.testing.assert_close((c.force_origin_b-c.nominal_hand_b).norm(dim=-1), distances)

    def test_body_sampler_covers_single_and_both_hand_scenarios(self):
        c = following_command()
        c.force_body_sampling = "balanced_hands"
        c.eval_force_body_mask[:] = True
        torch.manual_seed(61)
        sample = c._sample_active_mask(10000).squeeze(-1)
        count = sample.sum(-1)
        left_only = (sample[:, 4] & (count == 1)).float().mean().item()
        right_only = (sample[:, 5] & (count == 1)).float().mean().item()
        both_only = (sample[:, 4] & sample[:, 5] & (count == 2)).float().mean().item()
        self.assertTrue(.17 < left_only < .23)
        self.assertTrue(.17 < right_only < .23)
        self.assertTrue(.17 < both_only < .23)
        self.assertTrue(.17 < (count == 0).float().mean().item() < .23)
        c.eval_force_body_mask[:5] = False
        self.assertEqual(c._sample_active_mask(1000)[:, :5].sum().item(), 0.)

    def test_persistent_overload_reference_is_bounded_and_can_recover(self):
        c = following_command()
        c.hand_reference_limit = .25
        c.set_eval_stable_force(15., [1., 0., 0.])
        for _ in range(400):
            c.before_update()
        self.assertLessEqual((c.admit.x[0]-c.nominal_hand_b).norm(dim=-1).max().item(), .250001)
        self.assertAlmostEqual(c.admit.v[0, 0, 5, 0].item(), 0., places=5)
        c.set_eval_stable_force(0., [1., 0., 0.])
        for _ in range(200):
            c.before_update()
        self.assertLess((c.admit.x[0]-c.nominal_hand_b).norm(dim=-1).max().item(), .001)

    def test_conditional_metrics_do_not_mix_idle_success_into_tracking(self):
        c = following_command()
        c.force_enable[0, 5] = True
        c.root_vel_ref_w[0, 0] = .22
        c._accumulate_tracking_metrics()  # commanded but stationary: failure
        c.asset.data.root_lin_vel_w[0, 0] = .22
        c._accumulate_tracking_metrics()  # actual tracking: success
        c.root_vel_ref_w.zero_()
        c.asset.data.root_lin_vel_w.zero_()
        c._accumulate_tracking_metrics()  # idle must not increase success
        d = c.tracking_diagnostics()
        self.assertAlmostEqual(d['settle/moving_fraction'], 2/3, places=6)
        self.assertEqual(d['settle/moving_tracking_fraction'], .5)
        self.assertAlmostEqual(d['settle/moving_velocity_error_m_s'], .11, places=6)
        self.assertEqual(d['settle/rest_speed_m_s'], 0.)
        self.assertEqual(d['settle/moving_samples'], 2.)
        self.assertEqual(c.metric_sums.sum().item(), 0.)
        self.assertEqual(c.tracking_diagnostics()['settle/moving_samples'], 0.)

    def test_nominal_leg_pose_is_relaxed_only_during_translation(self):
        c = following_command()
        c.lower_joint_ids = torch.tensor([0, 1])
        c.asset.data.joint_pos = torch.zeros(1, 2)
        c.init_joint_pos = torch.zeros(1, 2)
        idle = c.stand_lower_body()
        c.root_vel_ref_w[0, 0] = .22
        self.assertAlmostEqual(c.stand_lower_body().item()/idle.item(), .1, places=6)

    def test_airtime_requires_touchdown_and_forward_progress(self):
        c = following_command()
        c.root_vel_ref_w[0, 0] = .22
        c.contact_sensor.data.last_air_time[:] = .3
        c.contact_sensor.compute_first_contact = lambda dt: torch.tensor([[True, False]])
        self.assertEqual(c.yield_step_air_time()[0].item(), 0.)
        c.asset.data.root_lin_vel_w[0, 0] = .22
        self.assertAlmostEqual(c.yield_step_air_time()[0].item(), .18, places=6)

    def test_replay_metrics_use_held_reference_and_record_contacts(self):
        c = following_command()
        reference = torch.tensor([.22, 0., 0.])
        c.root_vel_ref_w[0, 0] = -.22  # next command must not corrupt this step's comparison
        c.asset.data.root_lin_vel_w[0, 0] = .22
        c.contact_sensor.data.current_contact_time[:] = torch.tensor([[.1, 0.]])
        c.asset.data.body_lin_vel_w[0, 0, 0] = .03
        row = ForceHelpers._following_motion_fields(c, 0, reference)
        self.assertEqual(row['tracking_ok'], 1)
        self.assertAlmostEqual(row['velocity_error_xy_m_s'], 0.)
        self.assertEqual(row['left_shoulder_yaw_contact'], 1)
        self.assertEqual(row['right_shoulder_yaw_contact'], 0)
        self.assertAlmostEqual(row['left_shoulder_yaw_speed_xy_m_s'], .03, places=6)
        self.assertAlmostEqual(row['spring_offset_m'], .2, places=6)


if __name__ == '__main__':
    unittest.main()
