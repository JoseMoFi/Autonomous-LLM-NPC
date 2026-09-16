// move_to_and_pickup — built-in parameterised sub-plan.
//
// Moves the NPC to a known item location and picks it up.
// Accepts an item type and a desired quantity:
//
//   !move_to_and_pickup(ItemId, N)
//
// Variants:
//   Rule A — item location already known (item_at belief present):
//     → MoveTo(X, Y) then PickUp(ItemId), loop until has_item(ItemId) >= N.
//
//   Rule B — item location NOT known yet:
//     → ExploreArea to the item's spawn zone to trigger Search, then retry.
//
// Parameters:
//   ItemId — item type constant, e.g. "wheat"
//   N      — desired quantity (integer)

// ── Rule A: item position is known ─────────────────────────────────────────
// Guard: item_at(ItemId, X, Y)  →  we know where it is, go pick it up.
// After pickup we check inventory; if we still need more, loop this plan.

// Rule A0: item_at known but has_item belief not yet present (zero in inventory).
// Without this variant AgentSpeak cannot unify "Have" in Rule A and no plan fires.
+!move_to_and_pickup(ItemId, N) : item_at(ItemId, X, Y) & not has_item(ItemId, _) <-
    .moveto(X, Y);
    .pickup(ItemId);
    !move_to_and_pickup(ItemId, N).

+!move_to_and_pickup(ItemId, N) : item_at(ItemId, X, Y) & has_item(ItemId, Have) & Have < N <-
    .moveto(X, Y);
    .pickup(ItemId);
    !move_to_and_pickup(ItemId, N).

// Rule A2: already have enough — nothing to do.
+!move_to_and_pickup(ItemId, N) : has_item(ItemId, Have) & Have >= N <-
    true.

// ── Rule B: item location unknown but spawn zone is known ─────────────────
// Guard: not item_at(ItemId, _, _) & zone_center already bound → go there and search.
+!move_to_and_pickup(ItemId, N) : not item_at(ItemId, _, _) & item_spawn(ItemId, SpawnZone) & zone_center(SpawnZone, ZX, ZY) <-
    .moveto(ZX, ZY);
    .search(ItemId);
    !move_to_and_pickup(ItemId, N).

// Rule B2: spawn zone not yet discovered — explore the whole map first.
+!move_to_and_pickup(ItemId, N) : not item_at(ItemId, _, _) & item_spawn(ItemId, SpawnZone) <-
    .explorearea(SpawnZone);
    !move_to_and_pickup(ItemId, N).
