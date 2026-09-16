from __future__ import annotations

"""
test_smoke_action_semantics.py — tests de Iter 1.

Cubre:
- Creacion de BeliefSpec valido e invalido.
- Creacion de ActionContract minimo valido.
- validate_contract: contrato correcto -> sin errores.
- validate_contract: attempt_budget negativo -> error.
- validate_contract: attempt_budget cero -> error.
- validate_contract: attempt_budget no-int -> error.
- validate_contract: on_exhaustion desconocido -> error.
- validate_contract: name vacio -> error.
- ContractRegistry: registro y recuperacion.
- ContractRegistry: get sobre nombre no registrado -> None (no lanza).
- ContractRegistry: register con contrato invalido -> ValueError.
- ContractRegistry: all_names y len.
- ContractRegistry: sobreescritura idempotente del mismo nombre.
"""

import pytest

from protocol.action_semantics import (
    ON_EXHAUSTION_POLICIES,
    ActionContract,
    BeliefSpec,
    ContractRegistry,
    validate_contract,
)


# ---------------------------------------------------------------------------
# BeliefSpec
# ---------------------------------------------------------------------------


class TestBeliefSpec:
    def test_valido_minimo(self) -> None:
        b = BeliefSpec(functor="current_position", args=["X", "Y"])
        assert b.functor == "current_position"
        assert b.args == ["X", "Y"]

    def test_valido_sin_args(self) -> None:
        b = BeliefSpec(functor="idle")
        assert b.args == []

    def test_valido_con_nota(self) -> None:
        b = BeliefSpec(functor="has_item", args=["ItemId", "N"], note="N >= 1")
        assert b.note == "N >= 1"

    def test_functor_vacio_lanza(self) -> None:
        with pytest.raises(ValueError, match="vacio"):
            BeliefSpec(functor="")

    def test_functor_uppercase_lanza(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            BeliefSpec(functor="CurrentPosition")

    def test_functor_mixedcase_lanza(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            BeliefSpec(functor="item_At")


# ---------------------------------------------------------------------------
# ActionContract — construccion basica
# ---------------------------------------------------------------------------


class TestActionContractCreation:
    def test_minimo_valido(self) -> None:
        c = ActionContract(name="MoveTo")
        assert c.name == "MoveTo"
        assert c.attempt_budget == 1
        assert c.on_exhaustion == "goal_failed"
        assert c.requires == []
        assert c.guarantees_on_success == []
        assert c.may_observe == []
        assert c.invalidates == []
        assert c.consumes == []
        assert c.binds == []
        assert c.notes == ""

    def test_con_campos_completos(self) -> None:
        c = ActionContract(
            name="Search",
            attempt_budget=3,
            on_exhaustion="replan_goal",
            requires=[BeliefSpec("current_position", ["X", "Y"])],
            may_observe=[BeliefSpec("item_at", ["ItemId", "IX", "IY"])],
            binds=["IX", "IY"],
            notes="Observacional puro",
        )
        assert c.attempt_budget == 3
        assert c.on_exhaustion == "replan_goal"
        assert len(c.requires) == 1
        assert len(c.may_observe) == 1
        assert c.binds == ["IX", "IY"]

    def test_listas_independientes_entre_instancias(self) -> None:
        """Verifica que las listas por defecto no se comparten entre instancias."""
        c1 = ActionContract(name="A")
        c2 = ActionContract(name="B")
        c1.requires.append(BeliefSpec("x"))
        assert c2.requires == []


# ---------------------------------------------------------------------------
# validate_contract
# ---------------------------------------------------------------------------


class TestValidateContract:
    def _make(self, **kwargs) -> ActionContract:
        defaults = dict(name="TestAction", attempt_budget=1, on_exhaustion="goal_failed")
        defaults.update(kwargs)
        return ActionContract(**defaults)

    # --- valido ---

    def test_contrato_valido_sin_errores(self) -> None:
        c = self._make()
        assert validate_contract(c) == []

    def test_contrato_valido_todas_las_politicas(self) -> None:
        for policy in ON_EXHAUSTION_POLICIES:
            c = self._make(on_exhaustion=policy)
            assert validate_contract(c) == [], f"Fallo con politica '{policy}'"

    def test_contrato_valido_budget_grande(self) -> None:
        c = self._make(attempt_budget=99)
        assert validate_contract(c) == []

    # --- name ---

    def test_name_vacio_da_error(self) -> None:
        c = self._make(name="")
        errs = validate_contract(c)
        assert any("name" in e for e in errs)

    def test_name_espacios_da_error(self) -> None:
        c = self._make(name="  ")
        errs = validate_contract(c)
        assert any("name" in e for e in errs)

    # --- attempt_budget ---

    def test_budget_cero_da_error(self) -> None:
        c = self._make(attempt_budget=0)
        errs = validate_contract(c)
        assert any("attempt_budget" in e for e in errs)

    def test_budget_negativo_da_error(self) -> None:
        c = self._make(attempt_budget=-1)
        errs = validate_contract(c)
        assert any("attempt_budget" in e for e in errs)

    def test_budget_no_int_da_error(self) -> None:
        c = self._make(attempt_budget=1.5)  # type: ignore[arg-type]
        errs = validate_contract(c)
        assert any("attempt_budget" in e for e in errs)

    def test_budget_string_da_error(self) -> None:
        c = self._make(attempt_budget="1")  # type: ignore[arg-type]
        errs = validate_contract(c)
        assert any("attempt_budget" in e for e in errs)

    # --- on_exhaustion ---

    def test_on_exhaustion_desconocido_da_error(self) -> None:
        c = self._make(on_exhaustion="reintentar_siempre")
        errs = validate_contract(c)
        assert any("on_exhaustion" in e for e in errs)

    def test_on_exhaustion_vacio_da_error(self) -> None:
        c = self._make(on_exhaustion="")
        errs = validate_contract(c)
        assert any("on_exhaustion" in e for e in errs)

    # --- multiples errores acumulados ---

    def test_multiples_errores_acumulados(self) -> None:
        c = self._make(name="", attempt_budget=-5, on_exhaustion="malo")
        errs = validate_contract(c)
        assert len(errs) >= 3


# ---------------------------------------------------------------------------
# ContractRegistry
# ---------------------------------------------------------------------------


class TestContractRegistry:
    def _registry(self) -> ContractRegistry:
        return ContractRegistry()

    def _contract(self, name: str = "MoveTo", **kwargs) -> ActionContract:
        return ActionContract(name=name, **kwargs)

    # --- basicos ---

    def test_nuevo_registry_vacio(self) -> None:
        r = self._registry()
        assert len(r) == 0
        assert r.all_names() == []

    def test_get_nombre_inexistente_devuelve_none(self) -> None:
        r = self._registry()
        assert r.get("NoExiste") is None

    # --- register y get ---

    def test_register_y_get(self) -> None:
        r = self._registry()
        c = self._contract("MoveTo")
        r.register(c)
        assert r.get("MoveTo") is c

    def test_register_incrementa_len(self) -> None:
        r = self._registry()
        r.register(self._contract("MoveTo"))
        r.register(self._contract("Search", attempt_budget=3))
        assert len(r) == 2

    def test_all_names(self) -> None:
        r = self._registry()
        r.register(self._contract("MoveTo"))
        r.register(self._contract("Drop"))
        assert set(r.all_names()) == {"MoveTo", "Drop"}

    # --- sobreescritura ---

    def test_register_sobreescribe_mismo_nombre(self) -> None:
        r = self._registry()
        c1 = self._contract("MoveTo", attempt_budget=1)
        c2 = self._contract("MoveTo", attempt_budget=2)
        r.register(c1)
        r.register(c2)
        assert r.get("MoveTo") is c2
        assert len(r) == 1

    # --- contrato invalido ---

    def test_register_contrato_invalido_lanza_value_error(self) -> None:
        r = self._registry()
        c = ActionContract(name="", attempt_budget=-1, on_exhaustion="malo")
        with pytest.raises(ValueError, match="invalido"):
            r.register(c)

    def test_register_invalido_no_contamina_registry(self) -> None:
        r = self._registry()
        c_ok = self._contract("MoveTo")
        r.register(c_ok)
        c_bad = ActionContract(name="", attempt_budget=0)
        with pytest.raises(ValueError):
            r.register(c_bad)
        assert len(r) == 1
        assert r.get("MoveTo") is c_ok

    # --- instancia global ---

    def test_instancia_global_importable(self) -> None:
        from protocol.action_semantics import CONTRACT_REGISTRY
        # La instancia global existe y empieza vacia en Iter 1
        assert isinstance(CONTRACT_REGISTRY, ContractRegistry)
