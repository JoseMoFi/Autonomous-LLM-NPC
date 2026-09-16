// obtain_from_peer — built-in parameterised sub-plan (Fase 17).
//
// Pide un item a otro NPC y lo recoge cuando lo deja en el suelo:
//
//   !obtain_from_peer(Peer, Item, Qty)   (firma que se ofrece al LLM)
//   !obtain_from_peer(Item, Qty)         (compatibilidad: pregunta a cualquier peer)
//
// Macro de coordinación escrito a mano (brazo SUB). Solo se carga con
// coordinación activa; en ATOM no se carga y el LLM compone
// .ask_peer/.request_peer/.await_peer y la recogida con MoveTo/PickUp.
//
// Fase 17n: el piloto de cooperación mostró que el LLM lo llama con el peer
// delante, igual que collect_from_peer(Peer, Item, Qty). Solo existía la
// versión (Item, Qty) y fallaba con "no applicable plan for +!obtain_from_peer/3".
//
// Variants:
//   ya tengo la cantidad          → nada.
//   un peer ya dejó el item       → recogerlo (collect_from_peer).
//   hay un peer al que preguntar  → preguntar, pedir, esperar y recoger.
//
// La espera es larga (300 s): el otro NPC puede tener que planificar con el
// LLM y fabricar el item antes de entregarlo.

// ── Firma con el peer explícito ─────────────────────────────────────────────
+!obtain_from_peer(Peer, Item, Qty) : has_item(Item, Have) & Have >= Qty <-
    true.

+!obtain_from_peer(Peer, Item, Qty) : peer_item_available(Peer, Item, Qty, X, Y) <-
    !collect_from_peer(Peer, Item, Qty).

+!obtain_from_peer(Peer, Item, Qty) : peer(Peer, _) <-
    .ask_peer(Peer, can_make, Item);
    .request_peer(Peer, achieve_has_item, Item, Qty);
    .await_peer(Peer, achieve_has_item, 300);
    !collect_from_peer(Peer, Item, Qty).

// ── Compatibilidad: sin peer, pregunta al primero que conozca ───────────────
+!obtain_from_peer(Item, Qty) : has_item(Item, Have) & Have >= Qty <-
    true.

+!obtain_from_peer(Item, Qty) : peer_item_available(P, Item, Qty, X, Y) <-
    !collect_from_peer(P, Item, Qty).

+!obtain_from_peer(Item, Qty) : peer(P, _) <-
    !obtain_from_peer(P, Item, Qty).
