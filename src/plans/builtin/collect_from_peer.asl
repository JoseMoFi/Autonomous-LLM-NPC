// collect_from_peer — built-in parameterised sub-plan (Fase 12).
//
// Recoge una entrega que un peer avisó (Nivel 1 del intercambio: el peer ya
// hizo .drop en su posición y avisó con inform-done -> peer_item_available).
// Invocado desde la variante `delegate` de la familia achieve_has_item
// (src/llm/family_plan.py) tras un .await_peer exitoso.
//
//   !collect_from_peer(P, Item, Qty)
//
// Guard: peer_item_available(P, Item, Qty, X, Y) — lo assertea BeliefStore
// (apply_peer_done) cuando PeerCoordBehaviour recibe el inform-done con
// entrega física. Sin esa creencia, la acción no tiene dónde ir a buscar el
// item -> no hay variante de reintento aquí (si el peer no informó ubicación,
// el fallo ya se propagó por el timeout/failure de .await_peer aguas arriba).

+!collect_from_peer(P, Item, Qty) : peer_item_available(P, Item, Qty, X, Y) <-
    .moveto(X, Y);
    .pickup(Item).
