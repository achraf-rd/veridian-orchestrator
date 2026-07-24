"""Inter-agent payload transforms — ported from the frontend pipelineStore
(toAgent2Requirements / toAgent3Specs) so the orchestrator hands each agent
exactly the shape its tool expects.
"""

from typing import Any

_DEFAULT_ENV = {"map": "Town03", "weather": "clear", "lighting": "day"}
_DEFAULT_EGO = {"state": "system_active", "lane": "center", "initial_speed": 50, "set_speed": None}


def testable_requirements(refinement: dict) -> list[dict]:
    """The refined, testable requirements (supports old `requirements` key)."""
    return refinement.get("testable") or refinement.get("requirements") or []


def to_agent2_requirements(
    refinement: dict, requirement_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """RefinementResult.testable -> Agent 2's strict requirement shape.

    `requirement_ids` (optional): forward only these testable requirements (subset
    targeting, e.g. "make test cases from just req 1 and 3").
    """
    wanted = set(requirement_ids) if requirement_ids else None
    return [
        {
            "id": r.get("id"),
            "original": r.get("original", ""),
            "complexity": r.get("complexity", "LOW"),
            "conflict_flag": r.get("conflict_flag", False),
            "num_scenarios": r.get("num_scenarios", 1),
            "overlap_with": r.get("overlap_with", []),
            "status": "valid",
        }
        for r in testable_requirements(refinement)
        if wanted is None or r.get("id") in wanted
    ]


def deep_merge(base: Any, patch: Any) -> Any:
    """Recursively merge `patch` into `base`. Dicts merge key-by-key; anything else
    (scalars, lists) is replaced by `patch`. Never mutates the inputs. This is how a
    free-form scenario override (e.g. {ego_vehicle:{initial_speed:20}}) is applied on
    top of a resolved spec without the caller having to re-emit the whole payload."""
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for k, v in patch.items():
            out[k] = deep_merge(out.get(k), v) if k in out else v
        return out
    return patch


def to_agent3_specs(
    test_cases: list[dict],
    overrides: dict[str, dict] | None = None,
    scenario_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Agent 2 TestCase[] -> Agent 3 ScenarioSpec[].

    `scenario_ids` (optional): build only these test cases (subset targeting).
    `overrides` (optional): map of scenario_id -> free-form patch deep-merged into that
    scenario's spec (e.g. tweak ego_vehicle.initial_speed) before Agent 3 runs.
    """
    overrides = overrides or {}
    wanted = set(scenario_ids) if scenario_ids else None
    specs = []
    for tc in test_cases:
        sid = tc.get("scenario_id")
        if wanted is not None and sid not in wanted:
            continue
        sil = tc.get("sil_section") or {}
        covers = tc.get("covers_requirements", [])
        spec = {
            "scenario_id": sid,
            "covers_requirements": covers,
            "requirement_id": covers[0] if covers else None,
            "feature_under_test": tc.get("feature_under_test", ""),
            "complexity": tc.get("complexity", "MEDIUM"),
            "test_phase": tc.get("test_phase", "SIL"),
            "tags": tc.get("tags", []),
            "description": tc.get("description", ""),
            "preconditions": sil.get("preconditions", []),
            "environment": sil.get("environment", _DEFAULT_ENV),
            "ego_vehicle": sil.get("ego_vehicle", _DEFAULT_EGO),
            "actors": sil.get("actors", []),
            "test_steps": sil.get("steps", []),
        }
        if sid in overrides:
            spec = deep_merge(spec, overrides[sid])
        specs.append(spec)
    return specs
