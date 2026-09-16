// achieve_explore_zone — built-in parameterised sub-plan.
//
// Ensures the NPC knows the center of a zone.
// If the zone is already known, it succeeds immediately.
// Otherwise it explores that zone and retries until zone_center is present.

+!achieve_explore_zone(ZoneTag) : zone_center(ZoneTag, _, _) <-
    true.

+!achieve_explore_zone(ZoneTag) : not zone_center(ZoneTag, _, _) <-
    .explorearea(ZoneTag);
    !achieve_explore_zone(ZoneTag).