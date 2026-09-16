# Ablación de sub-planes — resultados (`coordination_builtins`)

Sesiones analizadas: **36** · commit(s): 3c435fd

## 1. Comprobación de manipulación

| Exp/Config | N | manipulación incorrecta | no verificable | llamadas a move_to_and_pickup/craft_item en planes |
|---|---|---|---|---|
| CO5/ATOM | 6 | 0 | 0 | 0 |
| CO5/DET | 6 | 0 | 0 | 0 |
| CO5/SUB | 6 | 0 | 0 | 48 |
| CO6/ATOM | 6 | 0 | 0 | 0 |
| CO6/DET | 6 | 0 | 0 | 0 |
| CO6/SUB | 6 | 0 | 0 | 36 |

_Correcto = en ATOM no se cargaron los sub-planes y la sesión arrancó con `builtin_subplans=false` (y al revés en SUB). En ATOM, una llamada a esos nombres es un sub-goal inventado por el LLM (el pipeline lo expande con el propio LLM), no el builtin._

## 2. Éxito (todos los goals del manifiesto verificados en creencias)

| Exp | SUB: éxito [IC95] | ATOM: éxito [IC95] | DET: éxito [IC95] | SUB: sin replan | ATOM: sin replan | DET: sin replan |
|---|---|---|---|---|---|---|
| CO5 | 6/6 [0.61, 1.00] | 5/6 [0.44, 0.97] | 6/6 [0.61, 1.00] | 6/6 | 0/6 | 6/6 |
| CO6 | 6/6 [0.61, 1.00] | 6/6 [0.61, 1.00] | 6/6 [0.61, 1.00] | 6/6 | 1/6 | 6/6 |

**Comparaciones por pares**: Fisher exacto (éxito) y Mann-Whitney U (p bilateral, todas las sesiones de cada brazo).

| Exp | par | éxito | Fisher p | MW llm_calls | MW n_act | MW duración |
|---|---|---|---|---|---|---|
| CO5 | SUB vs ATOM | 6/6 vs 5/6 | 1.000 | 0.003 | 0.003 | 0.005 |
| CO5 | SUB vs DET | 6/6 vs 6/6 | 1.000 | 0.001 | 1.000 | 0.005 |
| CO5 | ATOM vs DET | 5/6 vs 6/6 | 1.000 | 0.003 | 0.003 | 0.005 |
| CO6 | SUB vs ATOM | 6/6 vs 6/6 | 1.000 | 0.004 | 0.004 | 0.005 |
| CO6 | SUB vs DET | 6/6 vs 6/6 | 1.000 | 0.002 | 0.025 | 0.005 |
| CO6 | ATOM vs DET | 6/6 vs 6/6 | 1.000 | 0.003 | 0.002 | 0.005 |
| (todos) | SUB vs ATOM | 12/12 vs 11/12 | 1.000 | 0.000 | 0.000 | 0.000 |
| (todos) | SUB vs DET | 12/12 vs 12/12 | 1.000 | 0.000 | 0.467 | 0.000 |
| (todos) | ATOM vs DET | 11/12 vs 12/12 | 1.000 | 0.000 | 0.000 | 0.000 |

## 3. Coste (mediana [Q1, Q3])

| Exp/Config | N | llm_calls | t LLM (s) | n_act | replans | t hasta éxito (s) | duración sesión (s) |
|---|---|---|---|---|---|---|---|
| CO5/ATOM | 6 | 39 [32, 42] | 91.586 [71.189, 96.503] | 27.5 [23, 29] | 3 [2, 3] | 116.204 [94.547, 123.035] | 123.211 [102.302, 127.336] |
| CO5/DET | 6 | 1 [1, 1] | 4.287 [4.228, 4.321] | 15 [15, 15] | 0 [0, 0] | 35.1 [34.946, 39.161] | 39.435 [38.2, 42.602] |
| CO5/SUB | 6 | 12 [12, 12] | 23.8 [23.73, 23.889] | 15 [15, 15] | 0 [0, 0] | 55.482 [54.125, 57.182] | 59.506 [57.921, 60.625] |
| CO6/ATOM | 6 | 30.5 [30, 32] | 73.223 [72.114, 76.08] | 30 [26, 30] | 1 [1, 1] | 111.536 [109.882, 114.108] | 115.21 [113.727, 117.229] |
| CO6/DET | 6 | 1 [1, 1] | 4.362 [4.333, 4.394] | 19 [19, 19] | 0 [0, 0] | 41.096 [37.864, 41.261] | 44.233 [41.343, 45.47] |
| CO6/SUB | 6 | 16 [16, 16] | 30.872 [29.426, 31.254] | 20 [19, 20] | 0 [0, 0] | 66.146 [65.837, 66.275] | 70.108 [69.667, 70.255] |

## Coordinación entre NPCs

| Exp/Config | N | sesiones con entrega | peticiones (med.) | entregas (med.) | rechazos | timeouts de espera | arbitrajes | profundidad máx. |
|---|---|---|---|---|---|---|---|---|
| CO5/ATOM | 6 | 5 | 2 [2, 2] | 1 [1, 1] | 7 | 15 | 0 | 1 |
| CO5/DET | 6 | 6 | 1 [1, 1] | 1 [1, 1] | 0 | 0 | 0 | 1 |
| CO5/SUB | 6 | 6 | 1 [1, 1] | 1 [1, 1] | 0 | 0 | 0 | 1 |
| CO6/ATOM | 6 | 6 | 4 [4, 4] | 2 [2, 2] | 0 | 3 | 11 | 2 |
| CO6/DET | 6 | 6 | 2 [2, 2] | 2 [2, 2] | 0 | 0 | 6 | 2 |
| CO6/SUB | 6 | 6 | 2 [2, 2] | 2 [2, 2] | 0 | 0 | 6 | 2 |

## 4. Fracasos: causa principal y final de sesión

| Exp/Config | fracasos | causas principales | final de sesión |
|---|---|---|---|
| CO5/ATOM | 1 | failed_after_replans:action_failed:peer_refused:already_failed: 1 | idle: 6 |
| CO5/DET | 0 | — | idle: 6 |
| CO5/SUB | 0 | — | idle: 6 |
| CO6/ATOM | 0 | — | idle: 6 |
| CO6/DET | 0 | — | idle: 6 |
| CO6/SUB | 0 | — | idle: 6 |

## Notas de lectura

- **Éxito**: la sesión cierra TODOS los goals del manifiesto con `goal_belief_met=true`. Una sesión cortada por tiempo cuenta como fracaso (no se descarta).
- **Sin replan**: éxito sin ningún `goal_replan_required` (el primer plan bastó).
- **IC95** de Wilson. **Fisher** exacto bilateral sobre éxito/fracaso. **Mann-Whitney** con aproximación normal (corrección de empates y de continuidad). Con N≈8 por brazo la potencia es baja: un p alto NO demuestra igualdad.
- La fila **(todos)** mezcla experimentos de dificultad distinta: descriptiva, no sustituye a la comparación por experimento.
- **Causa principal** (por prioridad): `failed_after_replans:<motivo>` > `failed:<motivo>` > `no_applicable_variant` > `closed_without_belief` > `timeout` > `pipeline_error` > `unknown`. Las señales secundarias por sesión están en el CSV.
