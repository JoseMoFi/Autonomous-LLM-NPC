# Ablación de sub-planes — resultados (`ablation_long`)

Sesiones analizadas: **160** · commit(s): a398488

## 1. Comprobación de manipulación

| Exp/Config | N | manipulación incorrecta | no verificable | llamadas a move_to_and_pickup/craft_item en planes |
|---|---|---|---|---|
| A1/ATOM | 16 | 0 | 0 | 0 |
| A1/SUB | 16 | 0 | 0 | 16 |
| A2/ATOM | 16 | 0 | 0 | 0 |
| A2/SUB | 16 | 0 | 0 | 16 |
| A3/ATOM | 16 | 0 | 0 | 0 |
| A3/SUB | 16 | 0 | 0 | 48 |
| A4/ATOM | 16 | 0 | 0 | 0 |
| A4/SUB | 16 | 0 | 0 | 72 |
| A5/ATOM | 16 | 0 | 0 | 0 |
| A5/SUB | 16 | 0 | 0 | 64 |

_Correcto = en ATOM no se cargaron los sub-planes y la sesión arrancó con `builtin_subplans=false` (y al revés en SUB). En ATOM, una llamada a esos nombres es un sub-goal inventado por el LLM (el pipeline lo expande con el propio LLM), no el builtin._

## 2. Éxito (todos los goals del manifiesto verificados en creencias)

| Exp | SUB: éxito [IC95] | ATOM: éxito [IC95] | SUB: sin replan | ATOM: sin replan |
|---|---|---|---|---|
| A1 | 16/16 [0.81, 1.00] | 15/16 [0.72, 0.99] | 16/16 | 13/16 |
| A2 | 16/16 [0.81, 1.00] | 16/16 [0.81, 1.00] | 16/16 | 15/16 |
| A3 | 16/16 [0.81, 1.00] | 10/16 [0.39, 0.81] | 16/16 | 5/16 |
| A4 | 16/16 [0.81, 1.00] | 16/16 [0.81, 1.00] | 16/16 | 16/16 |
| A5 | 16/16 [0.81, 1.00] | 11/16 [0.44, 0.86] | 0/16 | 1/16 |

**Comparaciones por pares**: Fisher exacto (éxito) y Mann-Whitney U (p bilateral, todas las sesiones de cada brazo).

| Exp | par | éxito | Fisher p | MW llm_calls | MW n_act | MW duración |
|---|---|---|---|---|---|---|
| A1 | SUB vs ATOM | 16/16 vs 15/16 | 1.000 | 0.000 | 0.309 | 0.000 |
| A2 | SUB vs ATOM | 16/16 vs 16/16 | 1.000 | 0.000 | 1.000 | 0.000 |
| A3 | SUB vs ATOM | 16/16 vs 10/16 | 0.018 | 0.000 | 0.000 | 0.000 |
| A4 | SUB vs ATOM | 16/16 vs 16/16 | 1.000 | 0.000 | 0.348 | 0.000 |
| A5 | SUB vs ATOM | 16/16 vs 11/16 | 0.043 | 0.000 | 0.180 | 0.012 |
| (todos) | SUB vs ATOM | 80/80 vs 68/80 | 0.000 | 0.000 | 0.322 | 0.001 |

## 3. Coste (mediana [Q1, Q3])

| Exp/Config | N | llm_calls | t LLM (s) | n_act | replans | t hasta éxito (s) | duración sesión (s) |
|---|---|---|---|---|---|---|---|
| A1/ATOM | 16 | 5 [5, 5] | 12.989 [12.869, 13.041] | 4 [4, 4] | 0 [0, 0] | 26.599 [26.372, 27.411] | 30.603 [30.485, 31.871] |
| A1/SUB | 16 | 3 [3, 3] | 8.926 [8.867, 8.973] | 4 [4, 4] | 0 [0, 0] | 22.892 [22.561, 23.501] | 27.338 [25.869, 27.425] |
| A2/ATOM | 16 | 5 [5, 5] | 12.896 [12.867, 12.975] | 8 [8, 8] | 0 [0, 0] | 28.993 [28.182, 30.309] | 32.742 [32.002, 34.198] |
| A2/SUB | 16 | 3 [3, 3] | 8.88 [8.839, 8.928] | 8 [8, 8] | 0 [0, 0] | 24.501 [23.956, 25.501] | 28.883 [28.117, 28.977] |
| A3/ATOM | 16 | 19 [7, 25] | 43.776 [16.521, 55.668] | 24.5 [14, 31] | 2 [0, 3] | 50.249 [41.951, 75.323] | 79.651 [47.066, 87.296] |
| A3/SUB | 16 | 5 [5, 5] | 12.155 [12.097, 12.206] | 11 [11, 11] | 0 [0, 0] | 38.14 [36.754, 39.336] | 41.849 [40.468, 42.733] |
| A4/ATOM | 16 | 11 [11, 11] | 28.058 [27.919, 28.276] | 10 [10, 10] | 0 [0, 0] | 52.868 [52.391, 54.552] | 57.023 [56.129, 58.496] |
| A4/SUB | 16 | 7 [7, 7] | 20.078 [19.955, 20.125] | 10 [10, 10] | 0 [0, 0] | 45.747 [45.133, 46.546] | 50.004 [48.577, 50.096] |
| A5/ATOM | 16 | 19 [19, 25] | 46.208 [41.449, 56.326] | 51 [46.5, 54.5] | 2 [2, 3] | 119.629 [102.376, 130.581] | 120.019 [105.234, 129.382] |
| A5/SUB | 16 | 9 [9, 9] | 22.396 [22.297, 22.46] | 34 [34, 65] | 1 [1, 1] | 89.891 [88.556, 95.898] | 93.862 [92.276, 99.142] |

## 4. Fracasos: causa principal y final de sesión

| Exp/Config | fracasos | causas principales | final de sesión |
|---|---|---|---|
| A1/ATOM | 1 | failed_after_replans:ladder_stuck: 1 | idle: 16 |
| A1/SUB | 0 | — | idle: 16 |
| A2/ATOM | 0 | — | idle: 16 |
| A2/SUB | 0 | — | idle: 16 |
| A3/ATOM | 6 | failed_after_replans:ladder_stuck: 6 | idle: 16 |
| A3/SUB | 0 | — | idle: 16 |
| A4/ATOM | 0 | — | idle: 16 |
| A4/SUB | 0 | — | idle: 16 |
| A5/ATOM | 5 | failed_after_replans:ladder_stuck: 4, failed_after_replans:action_failed:InvalidArgs: 1 | idle: 16 |
| A5/SUB | 0 | — | idle: 16 |

## Notas de lectura

- **Éxito**: la sesión cierra TODOS los goals del manifiesto con `goal_belief_met=true`. Una sesión cortada por tiempo cuenta como fracaso (no se descarta).
- **Sin replan**: éxito sin ningún `goal_replan_required` (el primer plan bastó).
- **IC95** de Wilson. **Fisher** exacto bilateral sobre éxito/fracaso. **Mann-Whitney** con aproximación normal (corrección de empates y de continuidad). Con N≈8 por brazo la potencia es baja: un p alto NO demuestra igualdad.
- La fila **(todos)** mezcla experimentos de dificultad distinta: descriptiva, no sustituye a la comparación por experimento.
- **Causa principal** (por prioridad): `failed_after_replans:<motivo>` > `failed:<motivo>` > `no_applicable_variant` > `closed_without_belief` > `timeout` > `pipeline_error` > `unknown`. Las señales secundarias por sesión están en el CSV.
