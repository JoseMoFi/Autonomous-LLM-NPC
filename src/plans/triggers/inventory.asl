// inventory.asl — reactive triggers on inventory belief changes.
//
// Bodies use ONLY the supported reactive actions (Fase 0):
//   .log('mensaje')   → log INFO
//   adopt_goal(sig)    → adopt a new goal
// Guards are evaluated natively with GuardEvaluator before firing.
// Loaded by NPCAgent.setup() from plans/triggers/.

// React when the NPC acquires at least one unit of an item.
// The head variable N is bound to the current quantity; the guard N >= 1 is
// evaluated by GuardEvaluator against the live belief tuples.
+has_item(ItemId, N) : N >= 1 <-
    .log('item adquirido').

// Spontaneous behaviour (Fase 9): once the NPC has gathered enough wheat it
// spontaneously decides to bake bread. adopt_goal carries the success_condition
// 'has_item(bread, 1)', so the REACTIVE goal is belief-verified (it closes only
// when the belief holds), not closed "unverified". Materialises the deliberative
// reactivity that until now was wired but log-only.
+has_item(wheat, N) : N >= 2 <-
    adopt_goal(achieve_bake_bread, 'has_item(bread, 1)').
