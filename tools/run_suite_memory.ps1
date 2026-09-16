<#
run_suite_memory.ps1 - bateria de memoria de planes (suite memory_reuse).

Por cada repeticion, experimento (A4, A5) y brazo (SUB, ATOM) corre un PAR de
sesiones que comparten directorio de memoria:

  <brazo>_M1  primera sesion: memoria vacia, planifica con el LLM y guarda el plan
  <brazo>_M2  segunda sesion: carga la memoria de M1 y reusa el plan (sin pedirlo al LLM)

El orden de los brazos rota por repeticion (contrabalanceo, desfase inicial por
semilla); dentro de un par, M1 siempre va antes que M2. Cada sesion la lanza
run_experiment.ps1 (-Runs 1 -RunIndex <rep>), igual que run_suite.ps1. Al
terminar ejecuta tools\analyze_memory.py solo sobre las sesiones de esta tanda.

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_suite_memory.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File tools\run_suite_memory.ps1              (runs de la suite: 2 -> 16 sesiones)
  powershell -ExecutionPolicy Bypass -File tools\run_suite_memory.ps1 -Runs 8      (version larga: 8 pares por celda -> 64 sesiones, ~1,5-2 h)

Regla de reproducibilidad: como run_experiment.ps1, exige arbol git limpio
(ficheros trackeados) salvo -Force.
#>
param(
    [string]$Suite = "memory_reuse",
    [int]$Runs = 0,          # 0 = "runs" de la suite
    [int]$Seed = -1,         # -1 = "seed" de la suite
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoAnalysis,
    [string]$BuildExe = "",
    [double]$MaxHours = 0,   # >0: no empieza repeticiones nuevas pasadas estas horas
    [switch]$ShowWindow
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

$suitePath = "tools\experiments\suites\$Suite.json"
if (-not (Test-Path $suitePath)) { throw "No existe la suite: $suitePath" }
$suiteObj = Get-Content $suitePath -Raw | ConvertFrom-Json

$experiments = @($suiteObj.experiments)
$arms = @($suiteObj.arms)
$passes = @($suiteObj.passes)
$nReps = if ($Runs -gt 0) { $Runs } else { [int]$suiteObj.runs }
$seedValue = if ($Seed -ge 0) { $Seed } else { [int]$suiteObj.seed }

$maxSeconds = 0
foreach ($exp in $experiments) {
    $mPath = "tools\experiments\$exp.json"
    if (-not (Test-Path $mPath)) { throw "No existe el manifiesto: $mPath" }
    $m = Get-Content $mPath -Raw | ConvertFrom-Json
    $maxSeconds += [int]$m.timeout_s * $arms.Count * $passes.Count * $nReps
}

# Calendario: por repeticion y experimento, los brazos rotan; cada brazo corre su par M1 -> M2.
$rng = New-Object System.Random($seedValue)
$offsets = @{}
foreach ($exp in $experiments) { $offsets[$exp] = $rng.Next($arms.Count) }
$schedule = @()
for ($rep = 1; $rep -le $nReps; $rep++) {
    foreach ($exp in $experiments) {
        $shift = ($rep - 1 + $offsets[$exp]) % $arms.Count
        for ($j = 0; $j -lt $arms.Count; $j++) {
            $arm = $arms[($shift + $j) % $arms.Count]
            foreach ($pass in $passes) {
                $schedule += [PSCustomObject]@{
                    rep = $rep; experiment = $exp; arm = $arm; pass = $pass
                    config = "${arm}_${pass}"; position = $j + 1
                }
            }
        }
    }
}

$logDir = "logs\experiments\suite_$Suite"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$scheduleFile = Join-Path $logDir "schedule_$stamp.jsonl"

Write-Host ("[mem] {0}: {1} x {2} x {3} x {4} rep = {5} sesiones (seed {6})" -f `
    $Suite, ($experiments -join ","), ($arms -join "/"), ($passes -join "->"), $nReps, $schedule.Count, $seedValue)
Write-Host ("[mem] Cota superior si TODAS las sesiones agotaran su timeout: {0} h" -f `
    [math]::Round($maxSeconds / 3600.0, 1))
$k = 0
foreach ($item in $schedule) {
    $k++
    Write-Host ("[mem]   {0,3}. rep {1}  {2}/{3}" -f $k, $item.rep, $item.experiment, $item.config)
}

$sinceEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$t0 = Get-Date
$k = 0
$executed = 0
$currentRep = 0
foreach ($item in $schedule) {
    if ($MaxHours -gt 0 -and -not $DryRun -and $item.rep -ne $currentRep) {
        if (((Get-Date) - $t0).TotalHours -ge $MaxHours) {
            Write-Host ("[mem] MaxHours ({0} h) alcanzado: no se empieza la repeticion {1}." -f $MaxHours, $item.rep)
            break
        }
    }
    $currentRep = $item.rep
    $k++
    $elapsedMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host ""
    Write-Host ("[mem] ===== {0}/{1} (rep {2}) {3}/{4} -- {5} min transcurridos =====" -f `
        $k, $schedule.Count, $item.rep, $item.experiment, $item.config, $elapsedMin)

    if (-not $DryRun) {
        $record = [PSCustomObject]@{
            index = $k; rep = $item.rep; experiment = $item.experiment; arm = $item.arm
            pass = $item.pass; config = $item.config; position = $item.position
            seed = $seedValue; suite = $Suite; started_at = (Get-Date).ToString("o")
        }
        Add-Content -Path $scheduleFile -Value ($record | ConvertTo-Json -Compress)
    }

    $expArgs = @{
        Id       = $item.experiment
        Config   = $item.config
        Runs     = 1
        RunIndex = $item.rep
        SuiteId  = $Suite
    }
    if ($Force) { $expArgs["Force"] = $true }
    if ($DryRun) { $expArgs["DryRun"] = $true }
    if ($BuildExe) { $expArgs["BuildExe"] = $BuildExe }
    if ($ShowWindow) { $expArgs["ShowWindow"] = $true }
    & "$PSScriptRoot\run_experiment.ps1" @expArgs
    $executed++
}

$totalMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host ""
Write-Host "[mem] Terminado: $executed de $($schedule.Count) sesiones en $totalMin min."

if ($DryRun) {
    Write-Host "[mem] DRYRUN -- no se ha lanzado ninguna sesion."
    return
}
Write-Host "[mem] Calendario registrado en $scheduleFile"

if (-not $NoAnalysis) {
    $pyExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
    $outDir = "out\$Suite"
    Write-Host "[mem] Analizando sesiones de esta tanda -> $outDir"
    & $pyExe "tools\analyze_memory.py" "logs\sessions" "--suite" $Suite "--out" $outDir "--since-epoch" "$sinceEpoch"
    Write-Host "[mem] Resultados: $outDir\MEMORIA_RESULTADOS.md"
}
