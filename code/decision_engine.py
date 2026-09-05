"""
decision_engine.py -- the deterministic core of the triage agent.

Input:  a {slot_id: value} dict.
Output: a Decision naming the first protocol rule that fires, or a
        "not enough information yet" result.

Function of rules.yaml + the slot dict. This is the ONLY place a disposition is chosen.

Three-valued logic
------------------
Every atom evaluates to TRUE, FALSE, or UNKNOWN. An atom is UNKNOWN when the
slot it names is absent from the dict (value is None / missing). A rule fires
only when its `when` clause is definitively TRUE. This lets the state machine
call `evaluate()` after every new slot: as long as the result is
`disposition is None`, it keeps asking questions; the moment a rule is
definitively TRUE it stops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules.yaml")


class Tri(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


def _and(values: list[Tri]) -> Tri:
    if any(v is Tri.FALSE for v in values):
        return Tri.FALSE
    if any(v is Tri.UNKNOWN for v in values):
        return Tri.UNKNOWN
    return Tri.TRUE


def _or(values: list[Tri]) -> Tri:
    if any(v is Tri.TRUE for v in values):
        return Tri.TRUE
    if any(v is Tri.UNKNOWN for v in values):
        return Tri.UNKNOWN
    return Tri.FALSE


def _none_of(values: list[Tri]) -> Tri:
    """Exception clause. Optimistic on UNKNOWN.

    The protocol's "Exception:" clauses mean "apply this rule UNLESS the
    exception clearly holds". So the exception blocks the rule only when it is
    DEFINITELY true. An unknown exception is treated as not applying, which
    keeps the engine routing UP (the safe direction) when information is
    missing. `state_machine.py` is responsible for asking the exception
    question before the conversation ends, when time allows.
    """
    if any(v is Tri.TRUE for v in values):
        return Tri.FALSE
    return Tri.TRUE


# --- atom evaluation ------------------------------------------------------------
def _eval_atom(atom: list, slots: dict[str, Any]) -> Tri:
    slot_name, op, expected = atom[0], atom[1], atom[2]
    if slot_name not in slots or slots[slot_name] is None:
        return Tri.UNKNOWN
    actual = slots[slot_name]

    try:
        if op == "==":
            return Tri.TRUE if actual == expected else Tri.FALSE
        if op == "!=":
            return Tri.TRUE if actual != expected else Tri.FALSE
        if op == ">":
            return Tri.TRUE if actual > expected else Tri.FALSE
        if op == ">=":
            return Tri.TRUE if actual >= expected else Tri.FALSE
        if op == "<":
            return Tri.TRUE if actual < expected else Tri.FALSE
        if op == "<=":
            return Tri.TRUE if actual <= expected else Tri.FALSE
        if op == "in":
            return Tri.TRUE if actual in expected else Tri.FALSE
        if op == "contains":
            return Tri.TRUE if expected in (actual or []) else Tri.FALSE
        if op == "contains_any":
            actual_set = set(actual or [])
            return Tri.TRUE if actual_set & set(expected) else Tri.FALSE
        if op == "nonempty":
            is_nonempty = bool(actual)
            return Tri.TRUE if is_nonempty == bool(expected) else Tri.FALSE
    except TypeError:
        # e.g. numeric comparison against a non-number the extractor mis-filled
        return Tri.UNKNOWN

    raise ValueError(f"unknown operator: {op!r}")


def _atom_slot(clause: Any) -> str | None:
    if isinstance(clause, list) and clause and isinstance(clause[0], str):
        return clause[0]
    return None


def unresolved_slots(clause: Any, slots: dict[str, Any]) -> list[str]:
    """Slot names referenced in the POSITIVE part (all / any) of a clause whose
    value is still unknown. `none` (exception) atoms are excluded -- those are
    handled by `exception_slots`. Order preserved, de-duplicated."""
    out: list[str] = []

    def walk(node: Any) -> None:
        name = _atom_slot(node)
        if name is not None:
            if name not in slots or slots[name] is None:
                if name not in out:
                    out.append(name)
            return
        if isinstance(node, dict):
            for kind in ("all", "any"):
                for child in node.get(kind, []):
                    walk(child)

    walk(clause)
    return out


def exception_slots(when_clause: Any) -> list[str]:
    """Every slot name appearing inside the top-level `none` of a `when`
    clause -- i.e. the protocol 'Exception:' conditions for that rule."""
    out: list[str] = []

    def collect(node: Any) -> None:
        name = _atom_slot(node)
        if name is not None:
            if name not in out:
                out.append(name)
            return
        if isinstance(node, dict):
            for kind in ("all", "any", "none"):
                for child in node.get(kind, []):
                    collect(child)

    if isinstance(when_clause, dict):
        for child in when_clause.get("none", []):
            collect(child)
    return out


def _eval_clause(clause: Any, slots: dict[str, Any]) -> Tri:
    """A clause is either an atom (list) or a group dict with all/any/none."""
    if isinstance(clause, list) and clause and isinstance(clause[0], str):
        return _eval_atom(clause, slots)

    if isinstance(clause, dict):
        results: list[Tri] = []
        if "all" in clause:
            results.append(_and([_eval_clause(c, slots) for c in clause["all"]]))
        if "any" in clause:
            results.append(_or([_eval_clause(c, slots) for c in clause["any"]]))
        if "none" in clause:
            results.append(_none_of([_eval_clause(c, slots) for c in clause["none"]]))
        return _and(results)

    raise ValueError(f"malformed clause: {clause!r}")


# --- public API ---------------------------------------------------------------
@dataclass
class Decision:
    disposition: str | None            # None == not enough info yet
    rule_id: str | None = None
    rule_out: list[str] = field(default_factory=list)
    care_advice: list[int] = field(default_factory=list)
    first_aid: list[int] = field(default_factory=list)
    route_to: str | None = None
    triager_judgment: bool = False
    protocol_text: str = ""
    # every rule that is *definitely* satisfied, in priority order -- useful
    # for logging / eval even though only the first drives the disposition
    all_matches: list[str] = field(default_factory=list)
    pregnant_flag: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.disposition is not None


class RulesEngine:
    def __init__(self, rules_path: str = _RULES_PATH):
        with open(rules_path, "r", encoding="utf-8") as fh:
            self.spec = yaml.safe_load(fh)
        self.order: list[str] = self.spec["meta"]["disposition_order"]
        self._dispositions = {d["name"]: d["rules"] for d in self.spec["dispositions"]}
        # sanity: yaml order matches the declared disposition_order
        yaml_order = [d["name"] for d in self.spec["dispositions"]]
        if yaml_order != self.order:
            raise ValueError(
                f"disposition_order {self.order} != order in file {yaml_order}"
            )

    def iter_rules(self):
        """Yield (disposition_name, rule_dict) in protocol priority order."""
        for disp_name in self.order:
            for rule in self._dispositions[disp_name]:
                yield disp_name, rule

    def rule_by_id(self, rule_id: str) -> dict | None:
        for _, rule in self.iter_rules():
            if rule["id"] == rule_id:
                return rule
        return None

    def rule_states(self, slots: dict[str, Any]) -> dict[str, Tri]:
        """{rule_id: Tri} for every rule -- TRUE (fires), FALSE (dead), or
        UNKNOWN (still reachable, needs more slots)."""
        return {rule["id"]: _eval_clause(rule["when"], slots)
                for _, rule in self.iter_rules()}

    def evaluate(self, slots: dict[str, Any]) -> Decision:
        matches: list[tuple[str, dict]] = []
        for disp_name in self.order:
            for rule in self._dispositions[disp_name]:
                if _eval_clause(rule["when"], slots) is Tri.TRUE:
                    matches.append((disp_name, rule))

        pregnant = bool(slots.get("pregnant"))

        if not matches:
            return Decision(disposition=None, pregnant_flag=pregnant)

        disp_name, rule = matches[0]
        return Decision(
            disposition=disp_name,
            rule_id=rule["id"],
            rule_out=list(rule.get("rule_out", [])),
            care_advice=list(rule.get("care_advice", [])),
            first_aid=list(rule.get("first_aid", [])),
            route_to=rule.get("route_to"),
            triager_judgment=bool(rule.get("triager_judgment", False)),
            protocol_text=" ".join(rule.get("protocol", "").split()),
            all_matches=[r["id"] for _, r in matches],
            pregnant_flag=pregnant,
        )


_DEFAULT_ENGINE: RulesEngine | None = None


def evaluate(slots: dict[str, Any]) -> Decision:
    """Module-level convenience using a cached default engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RulesEngine()
    return _DEFAULT_ENGINE.evaluate(slots)


if __name__ == "__main__":
    demo = {
        "chest_pain_present_now": True,
        "duration": "over_5_min",
        "age": 58,
    }
    d = evaluate(demo)
    print(d)
