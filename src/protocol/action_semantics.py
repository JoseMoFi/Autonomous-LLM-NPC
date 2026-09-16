from __future__ import annotations

"""
action_semantics.py - fuente unica de verdad de contratos semanticos por accion.

Iter 1: schema + ContractRegistry vacio + validate_contract.
Iter 4: los siete contratos atomicos rellenan CONTRACT_REGISTRY.

Convenciones:
- Nombres de accion en PascalCase (como en PRIMITIVE_ACTIONS).
- Functores de beliefs en lowercase.
- on_exhaustion debe ser uno de ON_EXHAUSTION_POLICIES.
- Meta-predicados: "binding" (arg ya ligado) y "constant" (constante ground).
"""

from dataclasses import dataclass, field
from typing import Optional


ON_EXHAUSTION_POLICIES: frozenset[str] = frozenset(
    {"goal_failed", "replan_goal", "degrade_goal"}
)


@dataclass
class BeliefSpec:
    functor: str
    args: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.functor:
            raise ValueError("BeliefSpec.functor no puede ser vacio")
        if self.functor != self.functor.lower():
            raise ValueError(
                f"BeliefSpec.functor debe estar en lowercase: '{self.functor}'"
            )


@dataclass
class ActionContract:
    name: str
    attempt_budget: int = 1
    on_exhaustion: str = "goal_failed"
    requires: list[BeliefSpec] = field(default_factory=list)
    guarantees_on_success: list[BeliefSpec] = field(default_factory=list)
    may_observe: list[BeliefSpec] = field(default_factory=list)
    invalidates: list[BeliefSpec] = field(default_factory=list)
    consumes: list[BeliefSpec] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)
    notes: str = ""


def validate_contract(contract: ActionContract) -> list[str]:
    errors: list[str] = []
    if not contract.name or not contract.name.strip():
        errors.append("ActionContract.name no puede ser vacio")
    elif contract.name != contract.name.strip():
        errors.append(f"ActionContract.name contiene espacios: '{contract.name}'")
    if not isinstance(contract.attempt_budget, int):
        errors.append(f"ActionContract.attempt_budget debe ser int, no '{type(contract.attempt_budget).__name__}'")
    elif contract.attempt_budget < 1:
        errors.append(f"ActionContract.attempt_budget debe ser >= 1, recibido: {contract.attempt_budget}")
    if contract.on_exhaustion not in ON_EXHAUSTION_POLICIES:
        errors.append(f"ActionContract.on_exhaustion '{contract.on_exhaustion}' no es valido. Permitidos: {sorted(ON_EXHAUSTION_POLICIES)}")
    for fname in ("requires","guarantees_on_success","may_observe","invalidates","consumes","binds"):
        if getattr(contract, fname) is None:
            errors.append(f"ActionContract.{fname} no puede ser None")
    return errors


class ContractRegistry:
    def __init__(self) -> None:
        self._contracts: dict[str, ActionContract] = {}

    def register(self, contract: ActionContract) -> None:
        errors = validate_contract(contract)
        if errors:
            raise ValueError(f"Contrato '{contract.name}' invalido:\n" + "\n".join(f"  - {e}" for e in errors))
        self._contracts[contract.name] = contract

    def get(self, name: str) -> Optional[ActionContract]:
        return self._contracts.get(name)

    def all_names(self) -> list[str]:
        return list(self._contracts.keys())

    def __len__(self) -> int:
        return len(self._contracts)


CONTRACT_REGISTRY: ContractRegistry = ContractRegistry()


def _b(functor: str, *args: str, note: str = "") -> BeliefSpec:
    return BeliefSpec(functor=functor, args=list(args), note=note)


def _register_default_contracts(registry: ContractRegistry) -> None:
    registry.register(ActionContract(
        name="MoveTo", attempt_budget=1, on_exhaustion="goal_failed",
        requires=[_b("binding","X",note="X coord entera"), _b("binding","Y",note="Y coord entera")],
        guarantees_on_success=[_b("current_position","X","Y",note="Unity emite en ActionResult.payload")],
        invalidates=[_b("current_position",note="El anterior queda reemplazado")],
        notes="No observacional. Budget=1."
    ))
    registry.register(ActionContract(
        name="ExploreArea", attempt_budget=2, on_exhaustion="replan_goal",
        requires=[_b("constant","ZoneTag",note="ZoneTag debe ser constante conocida")],
        guarantees_on_success=[_b("current_position","X","Y",note="Unity emite current_position al cerrar")],
        may_observe=[
            _b("zone_center","ZoneTag","ZX","ZY",note="Puede descubrir la posicion de la zona"),
            _b("item_at","ItemId","IX","IY",note="Puede descubrir items en el area"),
        ],
        binds=["ZX","ZY"],
        notes="Observacional: mueve NPC, puede descubrir zone_center/item_at, no garantiza."
    ))
    registry.register(ActionContract(
        name="Search", attempt_budget=3, on_exhaustion="replan_goal",
        requires=[
            _b("constant","ItemId",note="ItemId debe ser constante"),
            _b("current_position","X","Y",note="NPC debe tener posicion conocida"),
        ],
        may_observe=[_b("item_at","ItemId","IX","IY",note="Puede descubrir el item; no garantizado")],
        binds=["IX","IY"],
        notes="Search NO mueve al NPC. NO produce zone_center. Observacional puro. Sin range."
    ))
    registry.register(ActionContract(
        name="PickUp", attempt_budget=1, on_exhaustion="goal_failed",
        requires=[
            _b("item_at","ItemId","X","Y",note="El item debe estar en el suelo"),
            _b("current_position","X","Y",note="NPC en la misma celda que el item"),
        ],
        guarantees_on_success=[_b("has_item","ItemId","N",note="Actualizado via InventoryUpdate")],
        invalidates=[_b("item_at","ItemId","X","Y",note="El item deja de estar en el suelo")],
        consumes=[_b("item_at","ItemId","X","Y",note="Presencia fisica consumida")],
        notes="No observacional. has_item llega via InventoryUpdate."
    ))
    registry.register(ActionContract(
        name="Craft", attempt_budget=1, on_exhaustion="goal_failed",
        requires=[
            _b("recipe_output","RecipeId","ZoneTag","ItemToCraft","1",note="Receta produce ItemToCraft"),
            _b("recipe","RecipeId","ZoneTag","IngredientItemId","QtyRequired",note="Ingrediente y cantidad"),
            _b("has_item","IngredientItemId","N",note="N >= QtyRequired"),
            _b("at_zone","ZoneTag",note="NPC must be physically inside the crafting zone"),
        ],
        guarantees_on_success=[_b("has_item","ItemToCraft","N",note="ItemToCraft +1 via InventoryUpdate")],
        invalidates=[_b("has_item","IngredientItemId",note="Cantidad anterior invalidada")],
        consumes=[_b("has_item","IngredientItemId","QtyRequired",note="QtyRequired unidades consumidas")],
        notes="Craft(ItemId, TargetId). ItemId = ingrediente principal; TargetId = recipeId o zona de crafteo. Produce 1 unidad del output de la receta. Requiere at_zone(ZoneTag). Budget=1."
    ))
    registry.register(ActionContract(
        name="Drop", attempt_budget=1, on_exhaustion="goal_failed",
        requires=[
            _b("has_item","ItemId","N",note="N >= 1"),
            _b("current_position","X","Y",note="Posicion actual determina donde cae el item"),
        ],
        guarantees_on_success=[
            _b("has_item","ItemId","N",note="Inventario -1 unidad"),
            _b("item_at","ItemId","X","Y",note="Item en el suelo en celda actual"),
        ],
        invalidates=[_b("has_item","ItemId",note="Cantidad anterior invalidada")],
        consumes=[_b("has_item","ItemId","1",note="Exactamente 1 unidad")],
        notes="Drop suelta exactamente 1 unidad. item_at derivado de current_position."
    ))
    registry.register(ActionContract(
        name="Wait", attempt_budget=1, on_exhaustion="goal_failed",
        requires=[_b("binding","Ticks",note="Numero de ticks a esperar")],
        notes="Accion neutra: consume tiempo, no produce ni invalida beliefs."
    ))


_register_default_contracts(CONTRACT_REGISTRY)
