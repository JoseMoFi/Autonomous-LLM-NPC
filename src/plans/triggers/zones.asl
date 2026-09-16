// zones.asl — reactive triggers on zone beliefs.
//
// Bodies use ONLY the supported reactive actions (Fase 0):
//   .log('mensaje')   → log INFO
//   adopt_goal(sig)    → adopt a new goal
// Any other action is rejected with a warning by BeliefStore._execute_trigger_body.
//
// Guards are evaluated natively with GuardEvaluator before firing.
// Loaded by NPCAgent.setup() from plans/triggers/.

// React when a new zone center is registered (zone discovered).
+zone_center(ZoneTag, _, _) : true <-
    .log('zona descubierta').

// at_zone(ZoneTag) is asserted by Unity (ZoneEntry) or derived from position.
// React on entry to leave a trace of the reactive lifecycle.
+at_zone(ZoneTag) : true <-
    .log('npc dentro de zona').
