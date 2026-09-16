// craft_item — built-in parameterised sub-plan.
//
// Moves the NPC to the recipe's crafting zone and crafts the item.
// The LLM emits: !craft_item(RecipeId, Qty)
//
// Variants:
//   Rule C  — already at the recipe zone: craft immediately.
//   Rule C2 — zone_center known but NPC not there yet: move then craft.
//   Rule C3 — zone not yet discovered: explore first, then retry.
//
// Parameters:
//   RecipeId — recipe constant, e.g. "bread_recipe"
//   Qty      — quantity to produce (integer)

// ── Rule C: already inside the crafting zone ────────────────────────────────
// Craft espera itemId = INGREDIENTE principal (no el output) y NO se pasa qty:
// el `qty` de Unity es la cantidad de INGREDIENTE de una hornada, y si se omite
// usa la cantidad declarada por la receta (verificado en UseActionHandler.cs).
// Pasar el Qty del goal (nº de outputs) rompía el match de receta. craft_item_check
// reintenta hasta acumular Qty outputs.
+!craft_item(RecipeId, Qty) : recipe(RecipeId, Zone, IngredientId, _) & at_zone(Zone) <-
    .craft(IngredientId, RecipeId);
    !craft_item_check(RecipeId, Qty).

// ── Rule C2: zone center known — move there first ───────────────────────────
+!craft_item(RecipeId, Qty) : recipe_output(RecipeId, Zone, ItemId, _) & zone_center(Zone, ZX, ZY) & not at_zone(Zone) <-
    .moveto(ZX, ZY);
    !craft_item(RecipeId, Qty).

// ── Rule C3: zone not yet discovered — explore first ───────────────────────
+!craft_item(RecipeId, Qty) : recipe_output(RecipeId, Zone, _, _) & not zone_center(Zone, _, _) <-
    .explorearea(Zone);
    !craft_item(RecipeId, Qty).

// ── Post-craft check: loop if not enough produced ───────────────────────────
+!craft_item_check(RecipeId, Qty) : recipe_output(RecipeId, _, ItemId, _) & has_item(ItemId, Have) & Have >= Qty <-
    true.

+!craft_item_check(RecipeId, Qty) : true <-
    !craft_item(RecipeId, Qty).
