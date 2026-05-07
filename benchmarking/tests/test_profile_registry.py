import unittest

from src.benchmark.runner import resolve_multi_turn_num_sessions
from src.workloads.profiles import filter_profiles, get_profile, resolve_profile_name


class ProfileRegistryTests(unittest.TestCase):
    def test_coding_agent_is_alias_for_coding_singleturn(self):
        self.assertEqual(resolve_profile_name("coding-agent"), "coding-singleturn")
        self.assertEqual(get_profile("coding-agent").name, "coding-singleturn")
        self.assertEqual(get_profile("coding-singleturn").dataset, "jsonl")

    def test_legacy_multiturn_profiles_remain_explicitly_runnable(self):
        self.assertEqual(get_profile("chat-multiturn-long").dataset, "sharegpt-multi-turn")
        self.assertEqual(get_profile("swebench-multiturn-short").dataset, "swebench-multi-turn")
        self.assertEqual(get_profile("terminalbench-multiturn-medium").dataset, "terminalbench-multi-turn")
        self.assertEqual(get_profile("osworld-multiturn-long").dataset, "osworld-multi-turn")

    def test_distributional_profiles_are_active_after_runner_wiring(self):
        active = set(filter_profiles(turn_style="multi-turn"))
        all_multi = set(filter_profiles(turn_style="multi-turn", include_inactive=True))

        canonical = {
            "chat-multiturn",
            "swebench-multiturn",
            "terminalbench-multiturn",
            "osworld-multiturn",
        }
        self.assertTrue(canonical.issubset(all_multi))
        self.assertTrue(canonical.issubset(active))

    def test_multi_turn_num_sessions_uses_concurrency_floor(self):
        profile = get_profile("osworld-multiturn")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            profile.num_sessions + 1,
        )

        self.assertEqual(effective, profile.num_sessions + 1)
        self.assertEqual(source, "concurrency_floor")

    def test_multi_turn_num_sessions_keeps_profile_default_when_large_enough(self):
        profile = get_profile("chat-multiturn")

        effective, source = resolve_multi_turn_num_sessions(
            profile,
            max(1, profile.num_sessions - 1),
        )

        self.assertEqual(effective, profile.num_sessions)
        self.assertEqual(source, "profile_default")

    def test_synthetic_profiles_carry_validated_apc_aware_settings(self):
        for name in (
            "chat-multiturn-synth",
            "swebench-multiturn-synth",
            "terminalbench-multiturn-synth",
            "osworld-multiturn-synth",
        ):
            with self.subTest(profile=name):
                profile = get_profile(name)
                self.assertEqual(profile.synthetic_filler_style, "code")
                self.assertEqual(profile.synthetic_target_chars_per_token, 3.8)
                self.assertTrue(profile.synthetic_prefix_aware)
                self.assertEqual(profile.synthetic_shared_prefix_tokens, 1024)

    def test_non_synthetic_distributional_profiles_do_not_force_apc_aware_settings(self):
        for name in (
            "chat-multiturn",
            "swebench-multiturn",
            "terminalbench-multiturn",
            "osworld-multiturn",
        ):
            with self.subTest(profile=name):
                profile = get_profile(name)
                self.assertEqual(profile.synthetic_filler_style, "")
                self.assertIsNone(profile.synthetic_target_chars_per_token)
                self.assertIsNone(profile.synthetic_prefix_aware)
                self.assertIsNone(profile.synthetic_shared_prefix_tokens)


if __name__ == "__main__":
    unittest.main()
