"""Config loading, override parsing, and the drafter != critic guarantee."""

from __future__ import annotations

import pytest
import yaml

from suitup.config import ConfigError, RoleBinding, load_config, parse_override

BASE = {
    "providers": {
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "api_key_env": "MISTRAL_API_KEY",
            "tier": "schema",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": None,
            "tier": "json_object",
        },
    },
    "roles": {
        "jd_analyst": {"provider": "mistral", "model": "mistral-small-latest"},
        "listing_selector": {"provider": "mistral", "model": "mistral-small-latest"},
        "bullet_selector": {"provider": "mistral", "model": "mistral-small-latest"},
        "drafter": {"provider": "mistral", "model": "mistral-small-latest"},
        "critic": {"provider": "mistral", "model": "mistral-large-latest"},
        "tagger": {"provider": "ollama", "model": "llama3.1"},
    },
}


def write_config(tmp_path, raw):
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(raw))
    return tmp_path


def test_loads_roles_and_providers(tmp_path):
    config = load_config(root=write_config(tmp_path, BASE))
    assert config.roles["drafter"].model == "mistral-small-latest"
    assert config.providers["ollama"].tier == "json_object"


def test_paths_derive_from_root(tmp_path):
    config = load_config(root=write_config(tmp_path, BASE))
    assert config.paths.listings == tmp_path / "library" / "listings"
    assert config.paths.configurations == tmp_path / "library" / "configurations.json"


def test_drafter_and_critic_must_differ(tmp_path):
    raw = {**BASE, "roles": {**BASE["roles"]}}
    raw["roles"]["critic"] = {"provider": "mistral", "model": "mistral-small-latest"}
    with pytest.raises(ConfigError, match="blind"):
        load_config(root=write_config(tmp_path, raw))


def test_role_routing_to_unknown_provider_is_rejected(tmp_path):
    raw = {**BASE, "roles": {**BASE["roles"]}}
    raw["roles"]["drafter"] = {"provider": "anthropic", "model": "whatever"}
    with pytest.raises(ConfigError, match="undefined provider"):
        load_config(root=write_config(tmp_path, raw))


def test_missing_role_is_rejected(tmp_path):
    raw = {**BASE, "roles": {k: v for k, v in BASE["roles"].items() if k != "critic"}}
    with pytest.raises(ConfigError, match="missing roles"):
        load_config(root=write_config(tmp_path, raw))


def test_unknown_role_is_rejected(tmp_path):
    raw = {**BASE, "roles": {**BASE["roles"], "sequencer": {"provider": "ollama", "model": "x"}}}
    with pytest.raises(ConfigError, match="unknown roles"):
        load_config(root=write_config(tmp_path, raw))


class TestParseOverride:
    def test_parses_role_provider_model(self):
        role, binding = parse_override("drafter=ollama:llama3.1")
        assert role == "drafter"
        assert binding == RoleBinding(provider="ollama", model="llama3.1")

    def test_model_names_containing_colons_survive(self):
        # Ollama tags look like `llama3.1:8b-instruct` — only the first colon separates.
        _, binding = parse_override("tagger=ollama:llama3.1:8b-instruct")
        assert binding.model == "llama3.1:8b-instruct"

    @pytest.mark.parametrize(
        "bad", ["drafter", "drafter=ollama", "drafter=:llama3", "drafter=ollama:"]
    )
    def test_malformed_overrides_rejected(self, bad):
        with pytest.raises(ConfigError):
            parse_override(bad)

    def test_unknown_role_rejected(self):
        with pytest.raises(ConfigError, match="unknown role"):
            parse_override("sequencer=ollama:llama3.1")


def test_override_applies_at_load(tmp_path):
    config = load_config(root=write_config(tmp_path, BASE), overrides=["drafter=ollama:llama3.1"])
    assert config.roles["drafter"].provider == "ollama"
    assert config.roles["drafter"].model == "llama3.1"


def test_override_can_break_the_blind_critique_guarantee(tmp_path):
    # An override is not a way around the invariant — it is revalidated.
    raw = {**BASE, "roles": {**BASE["roles"]}}
    raw["roles"]["critic"] = {"provider": "ollama", "model": "llama3.1"}
    with pytest.raises(ConfigError, match="blind"):
        load_config(root=write_config(tmp_path, raw), overrides=["drafter=ollama:llama3.1"])


def test_blind_critique_check_is_opt_out(tmp_path):
    # A quality guardrail, not a correctness one — the user can deliberately disable it.
    raw = {**BASE, "roles": {**BASE["roles"]}, "settings": {"require_blind_critique": False}}
    raw["roles"]["critic"] = {"provider": "mistral", "model": "mistral-small-latest"}
    config = load_config(root=write_config(tmp_path, raw))
    assert config.roles["drafter"] == config.roles["critic"]


class TestAnyRoleAnyProvider:
    """Every role must be routable to every provider — the core abstraction claim."""

    @pytest.mark.parametrize("role", sorted(BASE["roles"]))
    def test_each_role_can_move_to_any_provider(self, tmp_path, role):
        raw = {**BASE, "roles": {**BASE["roles"]}, "settings": {"require_blind_critique": False}}
        config = load_config(
            root=write_config(tmp_path, raw), overrides=[f"{role}=ollama:llama3.1"]
        )
        assert config.roles[role].provider == "ollama"

    def test_per_role_sampling_is_configurable(self, tmp_path):
        raw = {**BASE, "roles": {**BASE["roles"]}}
        raw["roles"]["drafter"] = {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "temperature": 0.9,
            "max_tokens": 2048,
        }
        config = load_config(root=write_config(tmp_path, raw))
        assert config.roles["drafter"].temperature == 0.9
        assert config.roles["drafter"].max_tokens == 2048

    def test_out_of_range_temperature_rejected(self, tmp_path):
        raw = {**BASE, "roles": {**BASE["roles"]}}
        raw["roles"]["drafter"] = {"provider": "mistral", "model": "m", "temperature": 5.0}
        with pytest.raises(Exception):
            load_config(root=write_config(tmp_path, raw))


class TestProfiles:
    def _raw_with_profiles(self):
        return {
            **BASE,
            "settings": {"require_blind_critique": False},
            "profiles": {
                "local": {
                    "drafter": {"provider": "ollama", "model": "llama3.1"},
                    "critic": {"provider": "ollama", "model": "llama3"},
                }
            },
        }

    def test_profile_overrides_defaults(self, tmp_path):
        config = load_config(
            root=write_config(tmp_path, self._raw_with_profiles()), profile="local"
        )
        assert config.roles["drafter"].provider == "ollama"
        assert config.roles["critic"].model == "llama3"

    def test_roles_absent_from_the_profile_fall_through(self, tmp_path):
        config = load_config(
            root=write_config(tmp_path, self._raw_with_profiles()), profile="local"
        )
        assert config.roles["jd_analyst"].provider == "mistral"

    def test_explicit_override_beats_the_profile(self, tmp_path):
        config = load_config(
            root=write_config(tmp_path, self._raw_with_profiles()),
            overrides=["drafter=mistral:mistral-large-latest"],
            profile="local",
        )
        assert config.roles["drafter"].provider == "mistral"

    def test_unknown_profile_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown profile"):
            load_config(root=write_config(tmp_path, self._raw_with_profiles()), profile="nope")

    def test_profile_naming_an_unknown_role_is_rejected(self, tmp_path):
        raw = self._raw_with_profiles()
        raw["profiles"]["local"]["sequencer"] = {"provider": "ollama", "model": "x"}
        with pytest.raises(ConfigError, match="unknown role"):
            load_config(root=write_config(tmp_path, raw), profile="local")

    def test_no_profile_leaves_defaults_untouched(self, tmp_path):
        config = load_config(root=write_config(tmp_path, self._raw_with_profiles()))
        assert config.roles["drafter"].provider == "mistral"
