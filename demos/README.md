# Demos

Cada fichero de esta carpeta define una demo: la build de Unity que se usa, los
objetivos de cada NPC y la configuración del sistema. Hay al menos una demo por
cada experimento de la evaluación ([`docs/EVALUACION.md`](../docs/EVALUACION.md)).

```powershell
run_demo.cmd -List           # lista las demos
run_demo.cmd -Demo A3        # ejecuta una demo
```

Antes hay que descargar el modelo (`ollama pull qwen3:8b`) y descomprimir
`builds.zip` de la última release en la raíz del repositorio. Mantén la ventana de
Unity en primer plano mientras se ejecuta, ya que la simulación se pausa al perder
el foco. Al cerrarla se detiene también el servidor, y la traza de la sesión queda
en `logs/sessions/`.

## Un agente

| Demo | Experimento | Qué se ve |
|---|---|---|
| `A1` | A1 | Un NPC va al campo, busca trigo, se desplaza hasta la casilla y lo recoge. |
| `A2` | A2 | Lo mismo, dos veces. |
| `A3` | A3 | Recolecta 2 trigos, va a la panadería y fabrica un pan. |
| `A4` | A4 | Dos objetivos encadenados: 2 trigos y después un pan. |
| `A5` | A5 | Tres panes: seis trigos y tres hornadas, volviendo al campo entre una y otra. |

## Cooperación

| Demo | Experimento | Qué se ve |
|---|---|---|
| `CO5` | CO5 | El panadero pide harina al molinero, que la fabrica y se la entrega. |
| `CO6` | CO6 | El molinero pide el pan al panadero, que a su vez le pide la harina (profundidad 2 con arbitraje). |

## Memoria de planes

Se ejecutan en orden. La primera planifica A4 con el modelo y guarda los
repertorios; la segunda los carga y resuelve los mismos objetivos sin pedir planes.

| Demo | Qué se ve |
|---|---|
| `MEM-1-aprender` | A4 planificado con el modelo; los repertorios se guardan en `plans/memory/runs/demo_memoria`. |
| `MEM-2-reutilizar` | A4 resuelto con la memoria anterior. |

## Arbitraje

Escenario de CO6 con planes deterministas, para que la única diferencia sea quién
decide cuando un agente queda esperando a otro.

| Demo | Qué se ve |
|---|---|
| `ARB-regla` | El conflicto lo resuelve la regla determinista. |
| `ARB-modelo` | El conflicto lo resuelve el modelo. |

## Configuraciones

Todas las demos pueden ejecutarse con cualquiera de las configuraciones de la
evaluación:

```powershell
run_demo.cmd -Demo A3                   # SUB: con planes auxiliares escritos a mano
run_demo.cmd -Demo A3 -NoSubplans       # ATOM: el modelo compone las acciones primitivas
run_demo.cmd -Demo CO5 -Deterministic   # DET: planes deterministas (solo demos de dos NPC)
```

## Crear una demo

Copia cualquiera de los JSON y cambia sus campos:

| Campo | Contenido |
|---|---|
| `title`, `description` | Texto que muestra `-List`. |
| `build` | `single` (un NPC, `npc_001`) o `coop` (`npc_miller` y `npc_baker`). |
| `npcs` | Número de NPC de la build. |
| `goals` | Objetivos por NPC: enunciado (`nl`) y condición de éxito (`condition`). |
| `env` | Variables `NPC_*` de configuración (ver `src/config/__init__.py`). |
| `memory` | `off`, `learn` o `reuse`, con `memory_group` como nombre del directorio compartido. |

Los objetos disponibles son `wheat`, `flour` y `bread`. En `single` el NPC sabe
fabricar pan a partir de trigo; en `coop` el molinero solo sabe fabricar harina a
partir de trigo y el panadero solo pan a partir de harina.
