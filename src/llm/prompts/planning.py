from __future__ import annotations

import re

from llm.catalogs import BELIEFS_LEGEND, ACTIONS_CATALOG, PEER_ACTIONS_CATALOG

def _format_entity_catalog(catalog: dict) -> str:
    """Returns a prompt block listing valid entity identifiers by semantic role.

    Empty string (no block injected) when the catalog is empty or absent.
    """
    if not catalog or not any(catalog.values()):
        return ""
    lines: list[str] = [
        "Entity vocabulary — use ONLY these exact identifiers as constants:",
    ]
    if catalog.get("zone_ids"):
        lines.append(f'  zone Tag        ∈ {{{", ".join(catalog["zone_ids"])}}}')
    if catalog.get("item_ids"):
        lines.append(f'  item ItemId     ∈ {{{", ".join(catalog["item_ids"])}}}')
    if catalog.get("delivery_tags"):
        lines.append(f'  delivery Tag    ∈ {{{", ".join(catalog["delivery_tags"])}}}')
    if catalog.get("recipe_ids"):
        lines.append(f'  recipe RecipeId ∈ {{{", ".join(catalog["recipe_ids"])}}}')
    return "\n".join(lines) + "\n"


# Sub-plan descriptions for LLM context (matched by prefix).
# Each entry: prefix → (args_signature, description)
# args_signature: shown after the sig name so the model knows how to call the sub-plan.
_SUBPLAN_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "move_to_and_pickup": (
        "(itemId, qty)",
        "find where the requested item can be obtained, navigate there, and pick up the required quantity",
    ),
    "craft_item": (
        "(recipeId, qty)",
        "navigate to the recipe's crafting zone (if needed) and craft the specified quantity of the output item",
    ),
    "achieve_explore_zone": (
        "(zoneTag)",
        "ensure that the target zone center is known; if unknown, explore until it is discovered",
    ),
    "achieve_gather":     ("", "gather a resource from the environment"),
    "achieve_craft":      ("", "craft an item using a recipe at the right zone"),
    "achieve_deliver":    ("", "deliver items to a target location"),
    "achieve_explore":    ("", "explore an area to discover zones or items"),
}


# Fase 17: sub-planes macro de coordinación (solo se etiquetan con coordinación;
# sin ella el prompt no cambia respecto a fases anteriores).
_COORDINATION_SUBPLAN_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "obtain_from_peer": (
        "(peerId, itemId, qty)",
        "ask another NPC that can make the item to produce it, wait for its delivery and pick it up",
    ),
    "collect_from_peer": (
        "(peerId, itemId, qty)",
        "go to where another NPC dropped the item and pick it up",
    ),
}


def _subgoal_label(sig: str, coordination: bool = False) -> str:
    """Return 'sig(args) — description' if a known description exists, else just 'sig'."""
    if coordination and sig in _COORDINATION_SUBPLAN_DESCRIPTIONS:
        args_sig, desc = _COORDINATION_SUBPLAN_DESCRIPTIONS[sig]
        return f"{sig}{args_sig} — {desc}"
    for prefix, (args_sig, desc) in _SUBPLAN_DESCRIPTIONS.items():
        if sig.startswith(prefix):
            label = f"{sig}{args_sig}" if args_sig else sig
            return f"{label} — {desc}"
    return sig


def _format_beliefs_for_contingency(beliefs: dict) -> str:
    """Format relevant beliefs as compact facts for the contingency prompt.

    Only includes functors useful for navigation / contingency reasoning.
    Returns empty string if beliefs are empty.
    """
    lines: list[str] = []
    for functor in ("zone_center", "item_spawn", "item_at", "has_item"):
        for row in beliefs.get(functor, []):
            args = ", ".join(str(a) for a in row)
            lines.append(f"  {functor}({args})")
    if not lines:
        return ""
    return (
        "\nCurrent beliefs (use these values — do NOT invent coordinates or names):\n"
        + "\n".join(lines) + "\n"
    )


def build_prompt(payload: dict, retry_errors: list[str] | None = None) -> tuple[str, str]:
    """
    Construye (prompt_usuario, prompt_sistema) para la tarea indicada.
    Si retry_errors no es None, añade el bloque de corrección al final del prompt.
    """
    task = payload.get("task", "")

    builders = {
        "parse_goals":      _build_parse_goals,
        "prioritize":       _build_prioritize,
        "arbitrate":        _build_arbitrate,
        # New v3 pipeline builders
        "step0_name":       _build_step0_name,
        "step1_success":    _build_step1_success,
        "step2_problem":    _build_step2_problem,
        # Legacy builders kept for compatibility
        "step0_reasoning":  _build_step0_reasoning,
        "step1_name":       _build_step1_name,
        "step2_guards":     _build_step2_guards,
        "step3_steps":      _build_step3_steps,
        "step3_repair":               _build_step3_repair,
        "step3_repair_completeness":  _build_step3_repair_completeness,
        "step5_need_plan":            _build_step5_need_plan,
    }

    builder = builders.get(task)
    if builder is None:
        return f"Unknown task: {task}", ""

    user_prompt, system_prompt = builder(payload)

    if retry_errors:
        error_block = "\n".join(f"  - {e}" for e in retry_errors)
        user_prompt += (
            f"\n\nThe previous response had the following errors. "
            f"Fix them and reply again:\n{error_block}"
        )

    return user_prompt, system_prompt


# ---------------------------------------------------------------------------
# Paso 0 — parse_goals: NL → sigs
# ---------------------------------------------------------------------------

def _build_parse_goals(p: dict) -> tuple[str, str]:
    goals_nl = p.get("goals_nl", [])

    system = (
        "Convert natural language objectives into BDI goal signatures (English snake_case).\n"
        "Reply ONLY with valid JSON. No explanation, no markdown, no code fences."
    )

    goals_text = "\n".join(f'  [{i}] "{g}"' for i, g in enumerate(goals_nl))
    user = f"""\
Convert these objectives into BDI goal signatures. Each objective is prefixed
with its index in square brackets:
{goals_text}

Rules for goal signatures:
- English snake_case only
- Pattern: achieve_<verb>_<object> (e.g. achieve_deliver_item_alpha, achieve_explore_zone_alpha)
- Translate any non-English words before naming

For each goal, also derive a success_condition: the minimal belief that must be TRUE
in the agent's belief base once the goal is accomplished.
Use ASL predicate format with real predicates from the domain (e.g. has_item(wheat, 1), knows_zone(farmland), item_at(wheat, X, Y)).
Use lowercase for all identifiers. Only reference entities explicitly mentioned in the objective.

CRITICAL:
- Do NOT use placeholders like belief(...), condition_met, zone_tag, inventory_alpha, status(flag).
- If combining multiple predicates, join them with " & " (ASL conjunction).
- Never put multiple predicates separated only by spaces.
- "source_index" MUST be the bracket index [i] of the objective this goal comes
  from. You may reorder, but source_index must always point back to the origin.

Valid examples:
- "has_item(wheat, 1)"
- "knows_zone(farmland)"
- "has_item(wheat, N) & N >= 1"

Reply with a JSON object containing a "goals" array:
{{
  "goals": [
    {{
      "sig": "achieve_verb_object",
      "source_index": 0,
      "priority": 1.0,
      "reason": "why this priority",
      "success_condition": "predicate(arg, ...)"
    }}
  ]
}}
"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 0b — prioritize: ordenar goals
# ---------------------------------------------------------------------------

def _build_prioritize(p: dict) -> tuple[str, str]:
    goals = p.get("goals", [])
    beliefs = p.get("beliefs", {})

    system = (
        "Rank BDI goals by priority given the current belief state.\n"
        "Reply ONLY with valid JSON. No explanation, no markdown."
    )

    goals_text = "\n".join(f'  - {g.get("sig")}' for g in goals)
    beliefs_text = str(beliefs)[:500]  # truncar para evitar tokens excesivos
    user = f"""\
Rank these goals by priority (1.0 = highest):
{goals_text}

Current beliefs (summary):
{beliefs_text}

Reply with:
[
  {{"sig": "achieve_...", "score": 0.9, "reason": "..."}},
  ...
]
"""
    return user, system


# ---------------------------------------------------------------------------
# arbitrate (Fase 14) — preempción: ¿seguir esperando al peer, o cambiar al
# goal delegado pendiente que ese MISMO peer me pidió?
# ---------------------------------------------------------------------------

def _build_arbitrate(p: dict) -> tuple[str, str]:
    current = p.get("current_goal", {})
    candidates = p.get("candidates", [])
    beliefs = p.get("beliefs", {})

    system = (
        "You are the decision module of an NPC's BDI agent. The NPC's current "
        "goal is blocked waiting for another NPC (a peer) to finish something. "
        "Meanwhile the NPC has accepted a request FROM that same peer (or from "
        "someone else) — a candidate goal it hasn't started yet, because this "
        "agent can only run one goal at a time.\n"
        "Decide whether to keep waiting, or switch to a candidate goal now.\n"
        "Key insight: if a candidate was requested BY the exact peer I'm "
        "waiting on, working on it may be exactly what unblocks that peer — "
        "and therefore unblocks me too. If no candidate relates to my wait, "
        "switching would not help and just delays my own goal for nothing.\n"
        "Reply ONLY with valid JSON. No explanation, no markdown."
    )

    candidates_text = "\n".join(
        f'  - sig="{c.get("sig")}" condition="{c.get("condition")}" '
        f'requested_by={c.get("requested_by")} goal_source={c.get("goal_source")}'
        for c in candidates
    ) or "  (none)"
    beliefs_text = str(beliefs)[:500]

    user = f"""\
Current (blocked) goal:
  sig="{current.get('sig')}" condition="{current.get('condition')}"
  waiting on peer: {current.get('waiting_on')}
  already waited: {current.get('waited_s')} s

Candidate goals I could switch to instead:
{candidates_text}

Current beliefs (summary):
{beliefs_text}

Reply with exactly one of:
{{"decision": "wait", "reason": "..."}}
{{"decision": "switch", "goal_sig": "<one of the candidate sigs above>", "reason": "..."}}
"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 0 — step0_name (v3): NL → sig + descripción detallada
# ---------------------------------------------------------------------------

def _build_step0_name(p: dict) -> tuple[str, str]:
    goal_nl = p.get("goal_nl", p.get("npc_statement", ""))
    npc_profile: dict = p.get("npc_profile") or {}

    system = """\
Convert a natural language objective into a BDI goal signature and a detailed description.
Reply ONLY with valid JSON. No explanation, no markdown, no code fences."""

    profile_block = ""
    if npc_profile:
        role = npc_profile.get("role", "")
        inventory = npc_profile.get("inventory", [])
        if role:
            profile_block += f"\nNPC role: {role}"
        if inventory:
            profile_block += f"\nNPC inventory: {', '.join(str(i) for i in inventory)}"
        profile_block += "\n"

    user = f"""\
Objective: "{goal_nl}"{profile_block}

Convert this into a BDI goal:

Rules for the goal signature:
- English snake_case only
- Pattern: achieve_<verb>_<object>
    Examples: achieve_deliver_item_alpha, achieve_collect_item_alpha, achieve_craft_item_beta
- Translate any non-English words

Rules for the description:
- One or two sentences, third person
- Describe WHAT the NPC must achieve and WHY it matters
- Be specific about items, quantities, and destinations if mentioned

Reply with:
{{
  "sig": "achieve_verb_object",
  "description": "The NPC must ... in order to ..."
}}"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 1 — step1_success (v3): sig + desc → success_conditions + .done variant
# ---------------------------------------------------------------------------

def _build_step1_success(p: dict) -> tuple[str, str]:
    sig = p.get("sig", "")
    description = p.get("description", "")
    catalog_block = _format_entity_catalog(p.get("entity_catalog", {}))

    system = """\
Define the success conditions for a BDI goal.
A success condition is a minimal ASL belief predicate that becomes TRUE when the goal is accomplished.
Reply ONLY with valid JSON. No explanation, no markdown, no code fences."""

    user = f"""\
Goal: {sig}
Description: "{description}"
{catalog_block}
Available belief predicates (for reference):
{BELIEFS_LEGEND}

Rules for success_conditions:
- List ONE or MORE ASL predicate expressions that must ALL be true for the goal to be done
- Use exact ASL syntax: functor(arg1, arg2) with lowercase identifiers
- Only reference entities explicitly mentioned in the description
- Prefer concise neutral predicates and explicit belief facts over narrative text
- The combined guard (ALL conditions ANDed) should capture the full "done" state
- Minimum 1 condition; maximum 3 (pick the most meaningful ones)

Rules for success_model:
- Return one entry per semantic variant that should generate its own execution branch
- Each entry must separate facts from guards
- facts are belief predicates and may bind variables
- guards are evaluable comparisons that depend on variables already bound in the same entry
- done_fragment must preserve fact-before-guard order for that entry only

Reply with:
{{
    "success_model": [
        {{
            "facts": ["belief(condition_x, V)"],
            "guards": ["V >= K"],
            "done_fragment": "belief(condition_x, V) & V >= K"
        }},
        {{
            "facts": ["belief(condition_y)"],
            "guards": [],
            "done_fragment": "belief(condition_y)"
        }}
  ],
    "success_conditions": [
        "belief(condition_x, V) & V >= K",
        "belief(condition_y)"
    ],
    "done_guard": "belief(condition_x, V) & V >= K & belief(condition_y)",
    "done_asl": "+!{sig} : belief(condition_x, V) & V >= K & belief(condition_y) <- true."
}}

IMPORTANT: condition_x, condition_y, V, K above are ABSTRACT PLACEHOLDERS.
Derive the actual predicate names, variable names, and threshold values from the goal description — do NOT use those placeholder names.

If there is only one success_model entry, done_guard equals that done_fragment directly (no extra & needed)."""
    return user, system


# ---------------------------------------------------------------------------
# Paso 2 — step2_problem (v3): neg_guard → problema NL + known_facts
# ---------------------------------------------------------------------------

def _build_step2_problem(p: dict) -> tuple[str, str]:
    sig = p.get("sig", "")
    description = p.get("description", "")
    neg_guard = p.get("neg_guard", "")
    facts = p.get("facts", [])
    guards = p.get("guards", [])
    bound_variables = p.get("bound_variables", [])
    beliefs: dict = p.get("beliefs") or {}
    catalog_block = _format_entity_catalog(p.get("entity_catalog", {}))
    beliefs_block = _format_beliefs_for_contingency(beliefs)

    system = """\
Describe in natural language what world state causes a specific belief condition to NOT be satisfied.
Reply ONLY with valid JSON. No explanation, no markdown, no code fences."""

    variant_lines: list[str] = []
    if facts:
        variant_lines.append("Variant facts:")
        variant_lines.extend(f"  - {fact}" for fact in facts)
    if guards:
        variant_lines.append("Variant guards:")
        variant_lines.extend(f"  - {guard}" for guard in guards)
    if bound_variables:
        variant_lines.append(f"Bound variables: {', '.join(str(v) for v in bound_variables)}")
    variant_block = "\n".join(variant_lines)
    if variant_block:
        variant_block = "\n" + variant_block + "\n"

    user = f"""\
Goal: {sig}
Description: "{description}"

The following condition is currently FAILING (not satisfied):
  {neg_guard}
{variant_block}{beliefs_block}{catalog_block}
Available beliefs (reference):
{BELIEFS_LEGEND}

Available primitive actions (reference only):
{ACTIONS_CATALOG}

Task:
1. Explain in plain English WHY this condition is failing (what the NPC lacks or cannot do).
   - Take into account the actions the NPC can actually perform.
2. List any KNOWN FACTS from the current beliefs that are relevant to solving this problem.
   - Use exact ASL predicate format: functor(arg1, arg2)
   - Only include facts that are currently TRUE (from the beliefs listed above)
   - If beliefs are empty, return an empty known_facts list

Reply with:
{{
    "problem_nl": "One or two sentences describing what is missing or impossible.",
    "known_facts": [
        "predicate(arg1, arg2)"
    ]
}}

IMPORTANT:
- problem_nl must describe the ACTUAL goal and beliefs, not a generic template.
- known_facts must be exact ASL predicates with REAL values from the beliefs above.
- Do NOT copy the schema text literally — replace every field with real content."""
    return user, system


# ---------------------------------------------------------------------------
# Paso 0 — step0_reasoning: razonamiento NL previo (Chain-of-Thought) [LEGACY]
# ---------------------------------------------------------------------------

def _build_step0_reasoning(p: dict) -> tuple[str, str]:
    goal_name = p.get("goal_name", "")
    npc_statement = p.get("npc_statement", "")
    existing_subgoals: list[str] = p.get("existing_subgoals", [])

    system = """\
Identify the minimal preconditions and action steps for a BDI agent goal.
Express everything in plain English — describe INTENT, not code.

Preconditions:
- State what might NOT be true when this goal starts.
- Use generic terms that describe the CONDITION, not specific names from the goal.
  Example: "the item location is not yet known", "the NPC does not have the required item".
- At least 1. Add more only when they are genuinely independent prerequisites.

Action steps:
- Describe WHAT to do, not which function to call.
    WRONG: "Search(item_alpha)", "MoveTo(X, Y)", "ExploreArea(zone_alpha)"
  RIGHT: "Search for the item in the current zone", "Move to the item's known location", "Roam the area to discover the spawn zone"
- Prefer reusing the listed sub-plans when they cover the task.
- At least 1 step. Add more only when the goal truly requires a sequence.

Reply ONLY with valid JSON. No explanation, no markdown, no code fences."""

    subgoal_lines = [f"  {_subgoal_label(sg)}" for sg in existing_subgoals] if existing_subgoals else ["  (none yet)"]
    subgoals_block = "\nAvailable sub-plans (reuse when they cover the task):\n" + "\n".join(subgoal_lines) + "\n"

    user = f"""\
Goal: {goal_name}
Objective: "{npc_statement}"
{subgoals_block}
Reply with:
{{
  "preconditions": [
    "one plain-English sentence per precondition"
  ],
  "action_plan": [
    "Step 1: one plain-English sentence describing what to do",
    "Step 2 (optional): ..."
  ]
}}"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 1 — step1_name: nombrar goal + npc_statement
# ---------------------------------------------------------------------------

def _build_step1_name(p: dict) -> tuple[str, str]:
    task_text = p.get("task_text", "")

    system = """\
Given an objective, produce:
1. A BDI goal name in English snake_case.
2. A one-sentence first-person description of the desired outcome.
Reply ONLY with valid JSON. No explanation, no markdown, no code fences."""

    user = f"""\
Objective: "{task_text}"

Rules for the goal name:
- English snake_case only
- Pattern: achieve_<verb>_<object> (optionally: achieve_<verb>_<object>_<aux>)
- Allowed prefixes: achieve, get, find, craft, deliver, flee, explore
- Translate any non-English words before naming

Reply with:
{{
  "goal": "achieve_verb_object",
  "npc_statement": "Needs to ... (third person or infinitive, full desired outcome, one sentence)"
}}"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 2 — step2_guards: guards + facts para un goal
# ---------------------------------------------------------------------------

def _build_step2_guards(p: dict) -> tuple[str, str]:
    goal_name = p.get("goal_name", "")
    npc_statement = p.get("npc_statement", "")
    reasoning = p.get("reasoning")
    catalog_block = _format_entity_catalog(p.get("entity_catalog", {}))

    system = """\
Define the GUARD CONDITIONS (preconditions) for a BDI goal.
Guards are conditions that must be TRUE before the goal is attempted.
They are evaluated in order and concatenated with AND (&).

Two kinds of conditions exist:
- FACTS: beliefs that exist or not in the knowledge base. No operator.
         Args can be constants (zone_alpha) or variables (N, X — uppercase).
- EVALUABLE GUARDS: numeric comparisons on variables already bound by a previous fact.
                    Always placed AFTER the fact that binds the variable.

There are NO boolean beliefs. Binary states are modeled as facts (present or absent).

Reply ONLY with valid JSON. No explanation, no markdown."""

    reasoning_block = ""
    if reasoning and isinstance(reasoning, dict):
        bullets = [b for b in reasoning.get("preconditions", []) if isinstance(b, str)][:5]
        if bullets:
            bullet_text = "\n".join(f"  - {b}" for b in bullets)
            reasoning_block = (
                f"\nContext (NL reasoning — reference only; do NOT add guards based on analogies):\n"
                f"{bullet_text}\n"
            )

    user = f"""\
Goal: {goal_name}
Objective: "{npc_statement}"
{reasoning_block}
Available beliefs (legend):
{BELIEFS_LEGEND}
{catalog_block}
Rules:
- facts list: one entry per belief needed, with constant or variable args as appropriate
- guards list: numeric comparisons only; each guard uses a variable bound by a previous fact
- Order matters: place each condition AFTER the one that binds its variables
- Add a "reason" field to each entry explaining why this condition is necessary
- ONLY add a guard if it is a DIRECT mechanical prerequisite of the goal (e.g. must know a zone to navigate to it, must have an item to deliver it). Do NOT add guards based on analogy or approximation from the context above.
- If a context bullet has no direct belief equivalent, skip it entirely — do not approximate it with a different entity.
- Scope: only use entity constants (items, zones) that are explicitly mentioned in the goal name or objective text. NEVER add facts for other items or zones from the catalog that are not directly part of this goal.

Reply with:
{{
  "facts": [
    {{"functor": "belief_name", "args": ["arg1", "ArgVar"], "reason": "why needed"}}
  ],
  "guards": [
        {{"expr": "QtyVar >= MinVar", "reason": "why needed"}}
  ]
}}

If no numeric guards are needed, return "guards": [].
If no facts are needed, return "facts": []."""
    return user, system


# ---------------------------------------------------------------------------
# Fase 16 — ablación de sub-planes: variante del prompt de step3 SIN
# move_to_and_pickup / craft_item. Sustituciones literales sobre el texto YA
# renderizado, de modo que el prompt por defecto (con sub-planes) queda byte a
# byte igual que antes. test_smoke_ablation_builtins comprueba que cada
# sustitución casa y que no queda ninguna mención.
# ---------------------------------------------------------------------------

_NO_BUILTIN_SUBPLAN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # system, modo no atómico
    (
        '    - If the need is "obtain item + quantity", and move_to_and_pickup is available,\n'
        '        prefer move_to_and_pickup(itemId, qty) instead of inventing a new achieve_get_* goal.\n',
        '',
    ),
    (
        '    WRONG — do NOT duplicate an existing acquisition plan:\n'
        '            {{"type":"subgoal","name":"achieve_verb_<item>","description":"Collect <item>"}}\n'
        '      when move_to_and_pickup(itemId, qty) is already available.\n',
        '',
    ),
    # system, modo atómico
    (
        'EXCEPTION: a plan whose LAST step is a terminal builtin sub-goal (move_to_and_pickup,\n'
        'craft_item, or achieve_explore_zone) is accepted even if it contains no other primitive,\n'
        'because these builtins already contain primitive actions internally and guarantee their output.\n',
        'EXCEPTION: a plan whose LAST step is the terminal builtin sub-goal achieve_explore_zone\n'
        'is accepted even if it contains no other primitive, because it already contains\n'
        'primitive actions internally and guarantees its output.\n',
    ),
    (
        '- Do NOT add a redundant Craft action after move_to_and_pickup — move_to_and_pickup already handles item acquisition.\n',
        '',
    ),
    # user, modo atómico
    (
        '   Pass its declared args: {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}\n',
        '   Pass its declared args: {"type": "subgoal", "name": "<sub_goal_name>", "args": [...]}\n',
    ),
    (
        '   Sub-goals guarantee their declared output (move_to_and_pickup → has_item; achieve_explore_zone → knows_zone).\n',
        '   Sub-goals guarantee their declared output (achieve_explore_zone → knows_zone).\n',
    ),
    (
        '10. Inventory beliefs (has_item) change only through PickUp, Drop, Craft, or move_to_and_pickup.\n',
        '10. Inventory beliefs (has_item) change only through PickUp, Drop, or Craft.\n',
    ),
    (
        '13. If the failing condition requires has_item, the plan must end with PickUp, Craft, Drop, or move_to_and_pickup.\n',
        '13. If the failing condition requires has_item, the plan must end with PickUp, Craft, or Drop.\n',
    ),
    (
        '        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]},\n'
        '        {"type": "action",  "name": "ActionName", "args": [...]}\n',
        '        {"type": "action",  "name": "ActionName", "args": [...]},\n'
        '        {"type": "action",  "name": "ActionName2", "args": [...]}\n',
    ),
    # user, modo no atómico
    (
        '3. For acquisition tasks (obtain item + qty), prefer move_to_and_pickup(itemId, qty) if present.\n',
        '3. For acquisition tasks (obtain item + qty), reuse a listed sub-goal only if one covers it.\n',
    ),
    (
        '8. Do not create synonyms/renames of existing sub-goals (e.g. achieve_get_X vs move_to_and_pickup).\n',
        '8. Do not create synonyms/renames of existing sub-goals.\n',
    ),
    (
        '        {"type": "subgoal", "name": "move_to_and_pickup", "args": ["<item_id>", <qty>]}\n',
        '        {"type": "subgoal", "name": "<existing_sub_goal>", "args": [...]}\n',
    ),
)


def _coordination_block(p: dict) -> str:
    """Fase 17: acciones de coordinación y NPCs conocidos (solo con coordinación LLM).

    Cadena vacía sin coordinación → el prompt queda byte a byte como antes.
    """
    if not p.get("coordination"):
        return ""
    peers = p.get("peers") or []
    peer_text = ", ".join(
        f"{row[0]} ({row[1]})" if len(row) > 1 else str(row[0])
        for row in peers if row
    ) or "(none)"
    return (
        "\nCoordination actions (messages to other NPCs; they never move the NPC; use type=\"action\"):\n"
        f"{PEER_ACTIONS_CATALOG}"
        f"Other NPCs: {peer_text}\n"
        "Coordination notes:\n"
        "  - A branch whose failing condition must be solved by another NPC may end with await_peer "
        "instead of PickUp, Craft or Drop.\n"
        "  - When await_peer succeeds, peer_item_available(NpcId, ItemId, Qty, X, Y) tells where the "
        "items were dropped; picking them up is a normal MoveTo + PickUp.\n"
    )


_COORDINATION_COMPLETENESS_RE = re.compile(
    r"^(13|19)\. If the failing condition requires has_item, the plan must end with (.+?)\.( .*)?$",
    re.MULTILINE,
)
_PRIMITIVE_GUARANTEES_LINE = (
    "   Primitive guarantees: PickUp/Craft/Drop → has_item; MoveTo/ExploreArea → current_position.\n"
)
_COORDINATION_GUARANTEES_LINE = (
    "   Coordination guarantees: request_peer → peer_promised (once the NPC agrees); "
    "await_peer → peer_item_available.\n"
)


def _with_coordination_completeness(text: str) -> str:
    """Fase 17n/17q: reglas de completitud (CRITICAL) con coordinación.

    17n: la regla exigía acabar en PickUp, Craft o Drop aunque el bloque de coordinación
    decía que la rama puede acabar en await_peer; el LLM recolectaba lo que solo otro
    NPC puede producir.
    17q (piloto de la 17p): los peldaños de pedir y esperar tienen como condición
    `peer_promised` / `peer_item_available`, que ninguna regla ni garantía mencionaba;
    el "otherwise" de la 17n les exigía acabar en PickUp/Craft/Drop y el baker
    crafteaba. Ahora cada predicado tiene su garantía, igual que has_item.
    Sin coordinación no se toca.
    """
    text = _COORDINATION_COMPLETENESS_RE.sub(
        lambda m: (
            f"{m.group(1)}. If the failing condition requires has_item, the plan must end with "
            f"{m.group(2)}, or with await_peer when another NPC must produce the item "
            f"(see Coordination notes).{m.group(3) or ''} If it requires peer_promised, end with "
            "request_peer; if it requires peer_item_available, end with await_peer."
        ),
        text,
    )
    return text.replace(
        _PRIMITIVE_GUARANTEES_LINE, _PRIMITIVE_GUARANTEES_LINE + _COORDINATION_GUARANTEES_LINE,
    )


# Fase 17r: en los peldaños peer_* la regla 1 ("PREFER a reusable sub-goal") hacía
# que el LLM añadiera achieve_explore_zone (el único sub-goal de ATOM) a casi todas
# las respuestas (piloto 17q). Hecho equivalente a la regla 7 de la 17q.
_PREFER_SUBGOAL_RULE_RE = re.compile(
    r"^1\. PREFER a reusable sub-goal[^\n]*\n   Pass its declared args:[^\n]*\n", re.MULTILINE,
)
_PEER_RULE1 = (
    "1. The reusable sub-goals above cover item acquisition or exploration, which do not change "
    "peer_* beliefs: they do not resolve this failing condition.\n"
)


def _without_builtin_subplans(text: str) -> str:
    """Quita del prompt de step3 las menciones a move_to_and_pickup/craft_item."""
    for old, new in _NO_BUILTIN_SUBPLAN_REPLACEMENTS:
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Paso 3 — step3_steps: secuencia de pasos del plan
# ---------------------------------------------------------------------------

def _build_step3_steps(p: dict) -> tuple[str, str]:
    goal_name = p.get("goal_name", "")
    npc_statement = p.get("npc_statement", "")
    existing_subgoals = p.get("existing_subgoals", [])
    catalog_block = _format_entity_catalog(p.get("entity_catalog", {}))

    # â”€â”€ v3 fields (new pipeline) â”€â”€
    neg_guard: str = p.get("neg_guard", "")
    problem_nl: str = p.get("problem_nl", "")
    known_facts: list = p.get("known_facts", [])

    # â”€â”€ v2 legacy fields â”€â”€
    facts = p.get("facts", [])
    guards = p.get("guards", [])
    bound_variables = p.get("bound_variables", [])
    reasoning = p.get("reasoning")
    belief_gap_hints: list[str] = p.get("belief_gap_hints", [])
    already_satisfied: list[str] = p.get("already_satisfied", [])
    unsatisfied_condition: str = p.get("unsatisfied_condition", "") or ""
    atomic_only: bool = bool(p.get("atomic_only", False))
    # Fase 16: ablación de sub-planes — sin menciones a move_to_and_pickup/craft_item.
    builtin_subplans: bool = bool(p.get("builtin_subplans", True))
    # Fase 17: coordinación planificada por el LLM (acciones de peer + NPCs conocidos).
    coordination: bool = bool(p.get("coordination", False))
    coordination_block = _coordination_block(p)
    _replan_hint: str = p.get("replan_hint", "") or ""
    replan_hint_block = (
        f"Previous attempt failed: {_replan_hint}. Adjust the plan accordingly.\n\n"
        if _replan_hint else ""
    )

    system = """\
Generate the ordered action steps to achieve a BDI goal.
Each step is either a primitive action or a sub-goal.

Sub-goal detection rule (IMPORTANT):
    Reuse-first policy:
    - If an existing sub-goal already solves the need, reuse it.
    - If the need is "obtain item + quantity", and move_to_and_pickup is available,
        prefer move_to_and_pickup(itemId, qty) instead of inventing a new achieve_get_* goal.

  If a group of steps requires obtaining a resource the NPC does NOT have
  guaranteed (e.g. must find, explore, pick up an item), group those steps
  into a SINGLE sub-goal step with:
    - type: "subgoal"
    - name: achieve_<verb>_<object>   (English snake_case)
    - description: what the sub-goal must accomplish (one sentence)
    - replaces_steps: list of NL strings describing the steps it replaces

  Only create a NEW sub-goal if all are true:
  - no existing sub-goal can represent that task,
  - the task is independently reusable,
  - and the task is not a trivial alias of the parent goal.

  Do NOT create a new sub-goal for trivial individual actions.

  WRONG — do NOT sub-goal a single atomic action:
        {{"type":"subgoal","name":"achieve_move_to_<zone>","description":"Move to <zone>"}}
    WRONG — do NOT duplicate an existing acquisition plan:
            {{"type":"subgoal","name":"achieve_verb_<item>","description":"Collect <item>"}}
      when move_to_and_pickup(itemId, qty) is already available.
  RIGHT — sub-goal a multi-step acquisition that can independently fail:
                {{"type":"subgoal","name":"achieve_verb_<item>","description":"Collect at least N units of <item> from <zone>.",
            "replaces_steps":["Go to <zone>","Search <item>","Pick it up"]}}

Reply ONLY with valid JSON. No explanation, no markdown."""

    if atomic_only:
        system = """\
Generate the ordered action steps to achieve a BDI goal.
Prefer reusable sub-goals from the provided list when they cover an acquisition or exploration task.
You MUST include at least one primitive action (type="action") — plans with ONLY sub-goal steps are rejected.
EXCEPTION: a plan whose LAST step is a terminal builtin sub-goal (move_to_and_pickup,
craft_item, or achieve_explore_zone) is accepted even if it contains no other primitive,
because these builtins already contain primitive actions internally and guarantee their output.

CRITICAL RULES:
- Do NOT emit replaces_steps.
- Each step must be either type="action" (primitive) or type="subgoal" (reuse from the provided list).
- Never invent sub-goal names not in the provided list.
- Do NOT add a redundant Craft action after move_to_and_pickup — move_to_and_pickup already handles item acquisition.

Reply ONLY with valid JSON. No explanation, no markdown."""

    subgoals_text = (
        "\n".join(f"  - {_subgoal_label(sg, coordination)}" for sg in existing_subgoals)
        if existing_subgoals else "  (none yet)"
    )

    # Build the context block depending on which fields are present
    if neg_guard:
        # v3 mode: neg_guard + problem_nl + known_facts
        known_facts_text = (
            "\n".join(f"  {f}" for f in known_facts)
            if known_facts else "  (none)"
        )
        vars_text = ", ".join(bound_variables) if bound_variables else "(none)"
        # Fase 17i: qué son las variables ligadas (antes solo se listaban sus nombres).
        bound_note = (
            "\n  These variables are ALREADY BOUND to concrete values when this branch runs: "
            "the Facts above hold for them. Use them directly as step arguments."
            if bound_variables else ""
        )
        variant_facts_text = (
            "\n".join(
                f"  {f['functor']}({', '.join(str(a) for a in f['args'])})"
                for f in facts
            )
            if facts else "  (none)"
        )
        variant_guards_text = (
            "\n".join(f"  {g['expr']}" for g in guards)
            if guards else "  (none)"
        )
        already_sat_text = (
            "\n".join(f"  - {c}" for c in already_satisfied)
            if already_satisfied else None
        )
        already_sat_block = (
            f"Already satisfied (guaranteed by guard — DO NOT generate steps for these):\n"
            f"{already_sat_text}\n\n"
            if already_sat_text else ""
        )
        # Use the specific unsatisfied atom when available; fall back to full guard.
        # This prevents the completeness rule from firing for unrelated atoms in the
        # compound guard (e.g. showing "not has_item(bread,1)" for the zone variant).
        _failing_shown = unsatisfied_condition if unsatisfied_condition else neg_guard
        _full_guard_line = (
            f"\nFull variant guard (for reference — DO NOT add steps for already-satisfied atoms):\n"
            f"  {neg_guard}\n"
            if unsatisfied_condition else ""
        )
        context_block = f"""\
{already_sat_block}Failing condition (the ONE sub-problem these steps must resolve):
  {_failing_shown}
{_full_guard_line}
Variant structure for this branch:
  Facts:
{variant_facts_text}
  Guards:
{variant_guards_text}
  Bound variables: {vars_text}{bound_note}

Why it is failing:
  {problem_nl}

Known relevant facts (use these values to reason — do NOT invent coordinates or names):
{known_facts_text}
"""
        if belief_gap_hints:
            gaps_text = "\n".join(f"  - {h}" for h in belief_gap_hints)
            context_block += f"""\
Missing beliefs / dependency notes for this branch:
{gaps_text}
"""
    else:
        # v2 legacy mode: facts + guards + bound vars
        facts_text = "\n".join(
            f"  {f['functor']}({', '.join(str(a) for a in f['args'])})" for f in facts
        )
        guards_text = "\n".join(f"  {g['expr']}" for g in guards)
        vars_text = ", ".join(bound_variables) if bound_variables else "(none)"
        reasoning_block = ""
        if reasoning and isinstance(reasoning, dict):
            bullets = [b for b in reasoning.get("action_plan", []) if isinstance(b, str)][:6]
            if bullets:
                bullet_text = "\n".join(f"  - {b}" for b in bullets)
                reasoning_block = (
                    f"\nContext (action sequence hints — use ONLY entities from the vocabulary above):\n"
                    f"{bullet_text}\n"
                )
        context_block = f"""\
Preconditions (already satisfied):
  Facts:
{facts_text}
  Guards:
{guards_text}
{reasoning_block}
Variables bound by guards: {vars_text}
"""
        if belief_gap_hints:
            gaps_text = "\n".join(f"  - {h}" for h in belief_gap_hints)
            context_block += f"""\
Missing beliefs / dependency notes for this branch:
{gaps_text}
"""

    # Fase 17i: con variables ligadas, las reglas 7/8/10 decían lo contrario al LLM
    # (zone_center con las mismas letras X, Y + Search; "no hardcodees / no inventes
    # coordenadas"). Sin variables ligadas el texto es idéntico al de antes.
    if bound_variables:
        _vars = ", ".join(bound_variables)
        rule7 = (
            "7. If zone_center(ZoneTag, ZX, ZY) is already known for the target zone, prefer MoveTo(ZX, ZY) "
            "and Search(itemId) to DISCOVER items there; do NOT use ExploreArea(ZoneTag) in that case. "
            "A zone center is not the position of a specific item."
        )
        rule8 = (
            f"8. Avoid hardcoding raw coordinates. The bound variables ({_vars}) are not hardcoded: they hold "
            "the values of the Facts of this branch — use them as arguments wherever the plan needs those "
            "values (e.g. MoveTo(X, Y) reaches the position given by a fact that binds X and Y)."
        )
        rule10 = (
            "10. Do not invent entity names, zones or coordinates "
            "(bound variables are not invented: the branch guard binds them)."
        )
    else:
        rule7 = (
            "7. If zone_center(ZoneTag, X, Y) is already known for the target zone, prefer MoveTo(X, Y) "
            "and Search(itemId); do NOT use ExploreArea(ZoneTag) in that case."
        )
        rule8 = "8. Avoid hardcoding raw coordinates unless the objective itself is explicitly about a fixed coordinate."
        rule10 = "10. Do not invent entity names, zones or coordinates."
    # Fase 17q: en los peldaños del protocolo (condición peer_*) la regla 7 empujaba a
    # MoveTo + Search. Se sustituye por el hecho equivalente a la regla 12.
    peer_failing = coordination and "peer_" in unsatisfied_condition
    if peer_failing:
        rule7 = (
            "7. The failing condition is about another NPC (peer_* beliefs), not about items or zones: "
            "MoveTo, Search, ExploreArea, PickUp and Craft do not change peer_* beliefs; "
            "only the coordination actions do."
        )

    if atomic_only:
        user = f"""\
Goal: {goal_name}
Objective: "{npc_statement}"

{context_block}
Available reusable sub-goals (prefer these when they cover the needed task):
{subgoals_text}

Available primitive actions:
{ACTIONS_CATALOG}
{catalog_block}{coordination_block}

Rules:
1. PREFER a reusable sub-goal from the list above when it covers an acquisition or exploration task.
   Pass its declared args: {{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}}
2. You MUST include at least one primitive action — plans with ONLY sub-goal steps are rejected.
3. If a sub-goal covers a multi-step acquisition, use it and add only the remaining primitives.
4. NEVER emit a sub-goal whose name is the current goal itself: "{goal_name}". That causes infinite recursion.
5. Only use sub-goal names that appear in the "Available reusable sub-goals" list above — never invent new ones.
6. If "Missing beliefs" are listed above, include discovery actions first (Search, ExploreArea, belief-derived navigation).
{rule7}
{rule8}
9. Steps must be executable and in correct order.
{rule10}

Completeness rules (CRITICAL — plan rejected if violated):
9. The final step must guarantee the goal predicate.
   Sub-goals guarantee their declared output (move_to_and_pickup → has_item; achieve_explore_zone → knows_zone).
   Primitive guarantees: PickUp/Craft/Drop → has_item; MoveTo/ExploreArea → current_position.
10. Inventory beliefs (has_item) change only through PickUp, Drop, Craft, or move_to_and_pickup.
11. Position beliefs (current_position) change only through MoveTo or ExploreArea.
12. Knowledge-only actions (Search, ExploreArea) update observational beliefs (item_at, zone_center); they do NOT place items in inventory and do NOT move the NPC to a specific item tile.
13. If the failing condition requires has_item, the plan must end with PickUp, Craft, Drop, or move_to_and_pickup.

{replan_hint_block}Reply with:
{{
    "steps": [
        {{"type": "subgoal", "name": "move_to_and_pickup", "args": ["wheat", 1]}},
        {{"type": "action",  "name": "ActionName", "args": [...]}}
    ]
}}"""
        if not builtin_subplans:
            user, system = _without_builtin_subplans(user), _without_builtin_subplans(system)
        if coordination:
            user = _with_coordination_completeness(user)
        if peer_failing:
            user = _PREFER_SUBGOAL_RULE_RE.sub(_PEER_RULE1, user, count=1)
        return user, system

    user = f"""\
Goal: {goal_name}
Objective: "{npc_statement}"

{context_block}
Available primitive actions:
{ACTIONS_CATALOG}
{catalog_block}{coordination_block}
Available sub-goals (already have plans — reuse when possible):
{subgoals_text}

Rules:
1. Steps must resolve the failing condition above (or achieve the goal if already satisfied).
2. REUSE an existing sub-goal when it covers the needed task (pass declared args).
3. For acquisition tasks (obtain item + qty), prefer move_to_and_pickup(itemId, qty) if present.
4. CREATE a new sub-goal (achieve_* name) only when no existing sub-goal matches the task.
    Include "description" and "replaces_steps".
5. A new sub-goal must define capability that is clearly different from all existing sub-goals.
6. Do NOT wrap a single primitive action in a sub-goal.
7. Steps must be in execution order; no redundant steps.
8. Do not create synonyms/renames of existing sub-goals (e.g. achieve_get_X vs move_to_and_pickup).
9. If reuse is possible, reuse wins over creating a new sub-goal.
10. NEVER emit a sub-goal whose name is exactly the current goal name: {goal_name}.
11. NEVER emit the same sub-goal name twice in the same plan.
12. If a candidate sub-goal would restate the parent goal, replace it with primitive actions or with a narrower reusable sub-goal.
13. CRITICAL: You MUST include at least one primitive action (MoveTo, Search, PickUp, etc.). Plans with ONLY sub-goal steps are INVALID.
14. If "Missing beliefs" are listed above, include the required discovery steps (Search, ExploreArea, MoveTo) BEFORE any step that needs that information.

Step types:
  Primitive action (single atomic operation from the catalog):
    {{"type": "action", "name": "MoveTo", "args": [X, Y]}}

  Existing sub-goal (reuse — pass declared args):
        {{"type": "subgoal", "name": "move_to_and_pickup", "args": ["<item_id>", <qty>]}}

  New sub-goal (multi-step independent acquisition group):
    {{
      "type": "subgoal",
        "name": "achieve_verb_<item>",
      "args": [],
            "description": "Collect at least N units of <item> from <zone>.",
            "replaces_steps": ["Go to <zone>", "Search <item>", "Pick it up"]
    }}

Replace them with the actual entity identifiers, zone tags, and quantities relevant to the current goal.

Completeness rules (CRITICAL — plan rejected if violated):
15. The final action of your sequence must guarantee the goal predicate (check guarantees_on_success in the catalog).
16. Inventory beliefs (has_item) change only through PickUp, Drop, or Craft. No other action produces inventory.
17. Position beliefs (current_position) change only through MoveTo or ExploreArea.
18. Knowledge-only actions (Search, ExploreArea) update observational beliefs (item_at, zone_center); they do NOT place items into inventory and do NOT move the NPC to a specific item tile.
19. If the failing condition requires has_item, the plan must end with PickUp, Craft, or Drop. If it requires current_position, end with MoveTo or ExploreArea.
{replan_hint_block}
Reply with:
{{
  "steps": [
    {{"type": "action", "name": "ActionName", "args": [...]}},
    {{
      "type": "subgoal",
            "name": "achieve_verb_<item>",
      "args": [],
            "description": "Collect at least N units of <item> from <zone>.",
            "replaces_steps": ["Go to <zone>", "Search <item>", "Pick it up"]
    }},
        {{"type": "action", "name": "ActionName2", "args": [...]}}
  ]
}}"""
    if not builtin_subplans:
        user, system = _without_builtin_subplans(user), _without_builtin_subplans(system)
    if coordination:
        user = _with_coordination_completeness(user)
    return user, system


# ---------------------------------------------------------------------------
# Paso 3b — step3_repair: corrección selectiva de steps fallidos
# ---------------------------------------------------------------------------

def _build_step3_repair(p: dict) -> tuple[str, str]:
    import json as _json
    goal_name = p.get("goal_name", "")
    all_steps = p.get("all_steps", [])
    failing_steps = p.get("failing_steps", [])  # [{index, step, errors}]
    bound_vars = p.get("bound_vars", [])
    fact_var_map = p.get("fact_var_map", {})      # {var → origin}
    facts = p.get("facts", [])
    guards = p.get("guards", [])
    existing_subgoals = p.get("existing_subgoals", [])
    reasoning = p.get("reasoning")

    system = """\
Fix flagged steps in a BDI plan.
Fix ONLY the steps listed as failing. Do NOT change any other step.
Return corrections in the exact format requested.
Reply ONLY with valid JSON. No explanation, no markdown."""

    # --- Contexto: plan completo con marcadores de error ---
    plan_lines: list[str] = []
    failing_indices = {item["index"] for item in failing_steps}
    for i, step in enumerate(all_steps):
        marker = " â† ERROR" if i in failing_indices else ""
        plan_lines.append(f"  [{i}] {_json.dumps(step)}{marker}")
    plan_text = "\n".join(plan_lines)

    # --- Detalle de errores ---
    error_lines: list[str] = []
    for item in failing_steps:
        error_lines.append(f"  Step {item['index']}: {_json.dumps(item['step'])}")
        for err in item.get("errors", []):
            error_lines.append(f"    ERROR: {err}")
    errors_text = "\n".join(error_lines)

    # --- Variables disponibles con origen ---
    if fact_var_map:
        vars_text = "\n".join(f"  {v}: bound by {origin}" for v, origin in fact_var_map.items())
    else:
        vars_text = "  (none)"

    # --- Contexto de reasoning si existe ---
    reasoning_block = ""
    if reasoning and isinstance(reasoning, dict):
        bullets = [b for b in reasoning.get("action_plan", []) if isinstance(b, str)][:4]
        if bullets:
            reasoning_block = (
                "\nOriginal NL reasoning (soft guidance):\n"
                + "\n".join(f"  - {b}" for b in bullets)
                + "\n"
            )

    subgoals_text = (
        "\n".join(f"  - {sg}" for sg in existing_subgoals)
        if existing_subgoals else "  (none yet)"
    )

    user = f"""\
Goal: {goal_name}

Full plan (steps in order):
{plan_text}
{reasoning_block}
Errors to fix:
{errors_text}

Available bound variables (use these — do NOT invent new uppercase variables):
{vars_text}

Available primitive actions:
{ACTIONS_CATALOG}

Available sub-goals:
{subgoals_text}

Rules:
- Each arg of a primitive action must match its signature (see catalog).
- Uppercase args are ASL variables and MUST appear in "Available bound variables" above.
- Lowercase args are constants (item names, zone tags, etc.) — use them as-is.
- Do NOT create new variables not listed above.

Return ONLY the corrected steps (index + corrected step):
{{
  "fixes": [
    {{"index": 2, "step": {{"type": "action", "name": "MoveTo", "args": ["X", "Y"]}}}},
        {{"index": 4, "step": {{"type": "subgoal", "name": "achieve_verb_object", "args": []}}}}
  ]
}}"""
    return user, system


# ---------------------------------------------------------------------------
# Paso 5 (V4 draft) — step5_need_plan: generar sub-plan para necesidad abierta
# ---------------------------------------------------------------------------

def _build_step5_need_plan(p: dict) -> tuple[str, str]:
        goal_name = p.get("goal_name", "")
        need_kind = p.get("need_kind", "")
        need_payload = p.get("need_payload", {})
        need_rationale = p.get("need_rationale", "")
        existing_subgoals = p.get("existing_subgoals", [])
        catalog_block = _format_entity_catalog(p.get("entity_catalog", {}))

        existing_text = (
                "\n".join(f"  - {sg}" for sg in existing_subgoals if isinstance(sg, str))
                if existing_subgoals else "  (none)"
        )

        system = """\
Generate ONE new reusable sub-goal to solve a specific unmet planning need.

Rules:
- Reply ONLY valid JSON.
- Return exactly one subgoal object.
- The subgoal must be narrower than the parent goal.
- Do NOT return the parent goal name.
- Do NOT return a name that already exists in existing_subgoals.
- Use English snake_case: achieve_<verb>_<object>.
"""

        user = f"""\
Parent goal:
    {goal_name}

Unmet need:
    kind: {need_kind}
    payload: {need_payload}
    rationale: {need_rationale}

Existing sub-goals (cannot reuse for this need):
{existing_text}

Available primitive actions:
{ACTIONS_CATALOG}
{catalog_block}

Return:
{{
    "subgoal": {{
        "sig": "achieve_verb_object",
        "description": "One sentence describing what capability this sub-goal provides.",
        "args": ["optional", "args"]
    }}
}}
"""
        return user, system


# ---------------------------------------------------------------------------
# step3_repair_completeness — repair cuando el plan no alcanza la condicion objetivo
# ---------------------------------------------------------------------------

def _build_step3_repair_completeness(p: dict) -> tuple[str, str]:
    """
    Prompt de repair cuando el simulador detecta que el plan no produce la
    success_condition. Recibe missing_predicates y un hint preformateado.
    No incluye ejemplos concretos (modelos pequeños los copian textualmente).
    """
    goal_sig          = p.get("goal_sig", "")
    success_condition = p.get("success_condition", "")
    current_steps     = p.get("current_steps", [])
    missing_preds     = p.get("missing_predicates", [])
    hint              = p.get("hint", "")
    catalog_block     = _format_entity_catalog(p.get("entity_catalog", {}))

    steps_text = "\n".join(
        f"  {i+1}. {s.get('type','?')} {s.get('name','?')}({', '.join(str(a) for a in (s.get('args') or []))})"
        for i, s in enumerate(current_steps)
    ) or "  (empty)"

    missing_text = "\n".join(f"  - {m}" for m in missing_preds) or "  (none listed)"

    system = """\
You are a BDI plan repair assistant. Your task is to extend or rewrite a plan \
so that its final action guarantees the goal belief condition.

Rules (CRITICAL — no exceptions):
1. The final action of the corrected plan MUST guarantee the goal condition \
   (check guarantees_on_success in the action catalog).
2. Inventory beliefs (has_item) change ONLY through PickUp, Drop, or Craft. \
   No other action produces inventory.
3. Position beliefs (current_position) change ONLY through MoveTo or ExploreArea.
4. Search and ExploreArea are observational: they update item_at or zone_center \
   but do NOT place items in inventory and do NOT guarantee a specific position.
5. Do NOT add steps that are already guaranteed by context. \
   Prefer the minimal extension.
6. Reply ONLY with valid JSON. No prose, no markdown, no extra keys.
"""

    user = f"""\
Goal: {goal_sig}
Success condition: {success_condition}

Current plan (does NOT satisfy the success condition):
{steps_text}

Missing predicates (not produced by the current plan):
{missing_text}

Repair hint:
{hint}

Available primitive actions:
{ACTIONS_CATALOG}
{catalog_block}

Return the corrected plan as a JSON list of steps. \
Each step is either an action or a subgoal:
{{
  "steps": [
    {{"type": "action",  "name": "ActionName", "args": {{"arg1": "val1"}}}},
    {{"type": "subgoal", "name": "achieve_verb_object", "args": []}}
  ]
}}
"""
    return user, system
