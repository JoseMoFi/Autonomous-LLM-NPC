from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def default_contracts_path() -> Path:
    src_root = Path(__file__).resolve().parents[2]
    return src_root / "plans" / "contracts" / "capability_contracts.json"


def _empty_contracts() -> dict:
    return {
        "version": 1,
        "contracts": {},
    }


def load_contracts(path: str | Path | None = None) -> dict:
    contracts_path = Path(path) if path else default_contracts_path()
    if not contracts_path.exists():
        return _empty_contracts()
    try:
        data = json.loads(contracts_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[CONTRACTS] Failed to read %s: %s", contracts_path, exc)
        return _empty_contracts()

    errors = validate_contracts(data)
    if errors:
        log.warning("[CONTRACTS] Invalid contracts in %s: %s", contracts_path, "; ".join(errors))
        return _empty_contracts()
    return data


def save_contracts(contracts: dict, path: str | Path | None = None) -> None:
    contracts_path = Path(path) if path else default_contracts_path()
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_contracts(contracts: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(contracts, dict):
        return ["contracts root must be an object"]

    version = contracts.get("version")
    if not isinstance(version, int):
        errors.append("version must be int")

    by_sig = contracts.get("contracts")
    if not isinstance(by_sig, dict):
        errors.append("contracts must be object")
        return errors

    for sig, spec in by_sig.items():
        if not isinstance(sig, str) or not sig.strip():
            errors.append("contract key must be non-empty string")
            continue
        if not isinstance(spec, dict):
            errors.append(f"contract '{sig}' must be object")
            continue
        provides = spec.get("provides", [])
        if not isinstance(provides, list):
            errors.append(f"contract '{sig}'.provides must be list")
            continue
        for idx, prov in enumerate(provides):
            if not isinstance(prov, dict):
                errors.append(f"contract '{sig}'.provides[{idx}] must be object")
                continue
            kind = prov.get("kind")
            if not isinstance(kind, str) or not kind.strip():
                errors.append(f"contract '{sig}'.provides[{idx}].kind must be non-empty string")
            constraints = prov.get("constraints", {})
            if not isinstance(constraints, dict):
                errors.append(f"contract '{sig}'.provides[{idx}].constraints must be object")

    return errors


def upsert_created_plan_contracts(contracts: dict, created_plans: list[dict]) -> bool:
    if not created_plans:
        return False

    by_sig = contracts.setdefault("contracts", {})
    changed = False

    if not isinstance(by_sig, dict):
        contracts["contracts"] = {}
        by_sig = contracts["contracts"]
        changed = True

    for item in created_plans:
        if not isinstance(item, dict):
            continue
        sig = str(item.get("sig", "")).strip()
        if not sig:
            continue

        provides = item.get("provides", [])
        if not isinstance(provides, list):
            provides = []
        normalized_provides: list[dict] = []
        for prov in provides:
            if not isinstance(prov, dict):
                continue
            kind = str(prov.get("kind", "")).strip()
            if not kind:
                continue
            constraints = prov.get("constraints", {})
            if not isinstance(constraints, dict):
                constraints = {}
            normalized_provides.append(
                {
                    "kind": kind,
                    "constraints": {str(k): v for k, v in constraints.items()},
                }
            )

        param_names = item.get("param_names", [])
        if not isinstance(param_names, list):
            param_names = []

        next_spec = {
            "description": str(item.get("description", "")).strip(),
            "provides": normalized_provides,
            "param_names": [str(name) for name in param_names if isinstance(name, str)],
            "source": "step5_need_plan",
        }

        if by_sig.get(sig) != next_spec:
            by_sig[sig] = next_spec
            changed = True

    return changed
