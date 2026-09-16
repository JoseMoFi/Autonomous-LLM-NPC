# Diseño de la evaluación

La evaluación comprueba si el modelo es capaz de construir planes ejecutables a
partir de las acciones primitivas del mundo y cómo de eficientes son esos planes
frente a los que escribiría un desarrollador.

## Configuraciones

La única variable entre SUB y ATOM es `NPC_BUILTIN_SUBPLANS`.

| Configuración | Descripción |
|---|---|
| **SUB** (referencia) | El modelo dispone de planes auxiliares escritos a mano para las acciones más complejas: `move_to_and_pickup`, `craft_item` y, en cooperación, `obtain_from_peer` y `collect_from_peer`. Están diseñados para resolver cada tarea con el mínimo número de pasos. |
| **ATOM** | Sin planes auxiliares ni mención a ellos en los prompts. El modelo compone directamente las acciones primitivas (`MoveTo`, `Search`, `PickUp`, `Craft`, `Drop`, `ask_peer`, `request_peer`, `await_peer`), guiado por la escalera de variantes que el pipeline deriva de los contratos de las acciones, de las recetas y del protocolo de coordinación. |
| **DET** (solo cooperación) | Planes de ambos agentes generados de forma determinista, sin el modelo. Sirve para comprobar que la coordinación funciona con independencia de la planificación. |

Las configuraciones se definen en [`tools/run_experiment.ps1`](../tools/run_experiment.ps1)
(`Set-ConfigEnv`) y [`tools/experiments/configs.json`](../tools/experiments/configs.json).

## Experimentos

Los manifiestos están en [`tools/experiments/`](../tools/experiments/).

| Id | NPC | Objetivo | Qué exige | N por configuración | Límite |
|---|---|---|---|---|---|
| A1 | 1 | `has_item(wheat, 1)` | ir al campo, buscar, desplazarse y recoger | 16 | 270 s |
| A2 | 1 | `has_item(wheat, 2)` | lo anterior dos veces | 16 | 270 s |
| A3 | 1 | `has_item(bread, 1)` | 2 trigos, ir a la panadería y fabricar | 16 | 270 s |
| A4 | 1 | `has_item(wheat, 2)` y `has_item(bread, 1)` | dos objetivos encadenados | 16 | 270 s |
| A5 | 1 | `has_item(bread, 3)` | 6 trigos y 3 hornadas | 16 | 270 s |
| CO5 | 2 | `npc_baker`: `has_item(bread, 1)` | el molinero fabrica y entrega la harina | 6 | 720 s |
| CO6 | 2 | `npc_miller`: `has_item(bread, 1)` | ayuda en ambos sentidos, profundidad 2 y arbitraje | 6 | 960 s |

En los experimentos A y CO la memoria de planes está desactivada, de modo que
cada objetivo se planifica desde cero. En cooperación el arbitraje se resuelve
con la regla determinista en las tres configuraciones.

La batería de memoria (`tools/experiments/suites/memory_reuse.json`) repite A4 y
A5 en SUB y ATOM por pares de sesiones que comparten directorio de memoria: la
primera planifica con el modelo y guarda el repertorio, y la segunda arranca con
él y lo reutiliza.

## Métricas

- **Éxito**: todos los objetivos del manifiesto se cierran con la condición de
  éxito verificada contra las creencias. Una sesión que agota el límite de tiempo
  cuenta como fallida.
- **Éxito sin replanificar**: el primer plan fue suficiente.
- **Coste**: llamadas al modelo, tiempo de modelo, acciones enviadas al mundo,
  replanificaciones y duración de la sesión.
- **Coordinación**: peticiones, entregas, rechazos, esperas agotadas, arbitrajes
  y profundidad de delegación.
- **Estadística**: intervalos de confianza de Wilson al 95 %, prueba exacta de
  Fisher para el éxito y U de Mann-Whitney para el coste. En la batería de
  memoria, Wilcoxon de rangos con signo sobre los pares.

## Validez

- Las configuraciones se ejecutan intercaladas y con el orden contrabalanceado
  por una semilla fija.
- El lanzador exige un árbol de git limpio y cada sesión registra su commit.
- Los contratos de capacidades se aíslan por sesión (`NPC_ISOLATE_CONTRACTS`),
  para que una sesión no contamine a la siguiente.
- Cada sesión comprueba que la manipulación es correcta: en ATOM no se cargan
  los planes auxiliares.

## Análisis

```powershell
python tools/analyze_ablation.py logs/sessions --suite ablation_long --out out/ablation_long
python tools/analyze_ablation.py logs/sessions --suite coordination_builtins --out out/coordination_builtins
python tools/analyze_memory.py logs/sessions --out out/memory_reuse
```

Los lanzadores de `tools/` ejecutan estos análisis automáticamente al terminar
cada tanda. Los resultados publicados están en [`results/`](../results/).
