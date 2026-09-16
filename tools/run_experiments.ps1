<#
run_experiments.ps1 — Matriz de experimentos de la Fase 5.

Ejecuta N runs de cada configuración (A-D) llamando a test_session.ps1 con los
flags adecuados. Cada run deja su sesión en logs/sessions/. Luego analizar con:
    python tools/analyze_sessions.py logs/sessions --md > DOC/eval_runs.md

Configs:
  A (completo)        : refinamiento on,  memoria on,  modelo por defecto
  B (sin refinamiento): refinamiento OFF, memoria on,  modelo por defecto
  C (sin memoria)     : refinamiento on,  memoria OFF, modelo por defecto
  D (modelo alt)      : refinamiento on,  memoria on,  -Model <alternativo>

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1 -Runs 5
  powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1 -Runs 5 -Configs A,C
  powershell ... -AltModel qwen2.5:7b   # para la config D

OJO: cada run abre Unity + Ollama y dura varios minutos. 5 runs x 4 configs es
~1-2 h. Reproducibilidad: mismo NPCProfile de Unity y mismos goals NL.
#>
param(
    [int]$Runs = 5,
    [string[]]$Configs = @("A", "B", "C", "D"),
    [string]$AltModel = "qwen2.5:7b",
    [int]$TimeoutSec = 600
)

$here = $PSScriptRoot
$session = Join-Path $here "test_session.ps1"

function Invoke-Config($name, $extraArgs) {
    Write-Host "==================================================================="
    Write-Host "[exp] Config $name  x$Runs runs  args: $($extraArgs -join ' ')"
    Write-Host "==================================================================="
    for ($i = 1; $i -le $Runs; $i++) {
        Write-Host "[exp] $name run $i/$Runs"
        & powershell -ExecutionPolicy Bypass -File $session -TimeoutSec $TimeoutSec @extraArgs
    }
}

if ($Configs -contains "A") { Invoke-Config "A" @() }
if ($Configs -contains "B") { Invoke-Config "B" @("-NoRefinement") }
if ($Configs -contains "C") { Invoke-Config "C" @("-MemoryOff") }
if ($Configs -contains "D") { Invoke-Config "D" @("-Model", $AltModel) }

Write-Host "[exp] Hecho. Analiza con: python tools/analyze_sessions.py logs/sessions --md"
