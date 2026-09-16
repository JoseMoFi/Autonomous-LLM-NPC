<#
run_suite.ps1 - Fase 16. Ejecuta una SUITE de experimentos
(tools\experiments\suites\<Suite>.json) intercalando brazos: en cada repeticion,
para cada experimento, corre TODAS las configs de la suite seguidas, y la que va
primero ROTA de una repeticion a la siguiente (contrabalanceo; el desfase inicial
de cada experimento sale de la semilla). Si la tanda se interrumpe, las
repeticiones ya completadas siguen balanceadas entre brazos.

Cada sesion la lanza run_experiment.ps1 (-Runs 1 -RunIndex <rep>), que usa
test_session.ps1 (Ollama + servidor Python + build de Unity). Al terminar,
ejecuta tools\analyze_ablation.py solo sobre las sesiones de ESTA tanda.

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_suite.ps1
  powershell -ExecutionPolicy Bypass -File tools\run_suite.ps1 -Runs 1     (piloto: 1 repeticion)
  powershell -ExecutionPolicy Bypass -File tools\run_suite.ps1 -DryRun     (solo calendario y argumentos)
  powershell -ExecutionPolicy Bypass -File tools\run_suite.ps1 -ShowWindow (ver el juego; se pausa sin foco)

Unity corre por defecto en segundo plano (-batchmode -nographics): sin ventana y
sin pausarse aunque se use el PC para otras cosas.

Regla de reproducibilidad: como run_experiment.ps1, exige arbol git limpio
(ficheros trackeados) salvo -Force.
#>
param(
    [string]$Suite = "ablation_builtins",
    [int]$Runs = 0,          # 0 = "runs" de la suite
    [int]$Seed = -1,         # -1 = "seed" de la suite
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoAnalysis,
    [string]$BuildExe = "",
    [double]$MaxHours = 0,   # >0: no empieza repeticiones nuevas pasadas estas horas (las hechas quedan balanceadas)
    [switch]$ShowWindow      # Unity con ventana (se pausa sin foco); por defecto segundo plano
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

$suitePath = "tools\experiments\suites\$Suite.json"
if (-not (Test-Path $suitePath)) { throw "No existe la suite: $suitePath" }
$suiteObj = Get-Content $suitePath -Raw | ConvertFrom-Json

$experiments = @($suiteObj.experiments)
$configs = @($suiteObj.configs)
$nReps = if ($Runs -gt 0) { $Runs } else { [int]$suiteObj.runs }
$seedValue = if ($Seed -ge 0) { $Seed } else { [int]$suiteObj.seed }

# Validar manifiestos ANTES de lanzar nada.
$maxSeconds = 0
foreach ($exp in $experiments) {
    $mPath = "tools\experiments\$exp.json"
    if (-not (Test-Path $mPath)) { throw "No existe el manifiesto: $mPath" }
    $m = Get-Content $mPath -Raw | ConvertFrom-Json
    foreach ($cfg in $configs) {
        if (@($m.configs) -notcontains $cfg) { throw "El manifiesto $exp no declara la config $cfg" }
    }
    $maxSeconds += [int]$m.timeout_s * $configs.Count * $nReps
}

# Calendario intercalado y CONTRABALANCEADO: la config que va primero rota en
# cada repeticion. Con N multiplo del numero de configs, cada config va primero
# el mismo numero de veces por experimento (con 2 configs y N=8: 4 y 4).
$rng = New-Object System.Random($seedValue)
$offsets = @{}
foreach ($exp in $experiments) { $offsets[$exp] = $rng.Next($configs.Count) }
$schedule = @()
for ($rep = 1; $rep -le $nReps; $rep++) {
    foreach ($exp in $experiments) {
        $shift = ($rep - 1 + $offsets[$exp]) % $configs.Count
        for ($j = 0; $j -lt $configs.Count; $j++) {
            $schedule += [PSCustomObject]@{
                rep        = $rep
                experiment = $exp
                config     = $configs[($shift + $j) % $configs.Count]
                position   = $j + 1
            }
        }
    }
}

$logDir = "logs\experiments\suite_$Suite"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$scheduleFile = Join-Path $logDir "schedule_$stamp.jsonl"

Write-Host ("[suite] {0}: {1} x {2} x {3} rep = {4} sesiones (seed {5})" -f `
    $Suite, ($experiments -join ","), ($configs -join "/"), $nReps, $schedule.Count, $seedValue)
Write-Host ("[suite] Cota superior si TODAS las sesiones agotaran su timeout: {0} h" -f `
    [math]::Round($maxSeconds / 3600.0, 1))
$k = 0
foreach ($item in $schedule) {
    $k++
    Write-Host ("[suite]   {0,3}. rep {1}  {2}/{3}" -f $k, $item.rep, $item.experiment, $item.config)
}

# Solo sesiones que empiecen despues de este instante entran en el analisis.
$sinceEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$t0 = Get-Date
$k = 0
$executed = 0
$currentRep = 0
foreach ($item in $schedule) {
    # -MaxHours: solo se decide al empezar una repeticion, asi las completadas
    # siguen contrabalanceadas entre brazos.
    if ($MaxHours -gt 0 -and -not $DryRun -and $item.rep -ne $currentRep) {
        if (((Get-Date) - $t0).TotalHours -ge $MaxHours) {
            Write-Host ("[suite] MaxHours ({0} h) alcanzado: no se empieza la repeticion {1}. Repeticiones completas: {2}." -f `
                $MaxHours, $item.rep, ($item.rep - 1))
            break
        }
    }
    $currentRep = $item.rep
    $k++
    $elapsedMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host ""
    Write-Host ("[suite] ===== {0}/{1} (rep {2}) {3}/{4} -- {5} min transcurridos =====" -f `
        $k, $schedule.Count, $item.rep, $item.experiment, $item.config, $elapsedMin)

    if (-not $DryRun) {
        $record = [PSCustomObject]@{
            index = $k; rep = $item.rep; experiment = $item.experiment; config = $item.config
            position = $item.position; seed = $seedValue; suite = $Suite
            started_at = (Get-Date).ToString("o")
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
Write-Host "[suite] Terminado: $executed de $($schedule.Count) sesiones en $totalMin min."

if ($DryRun) {
    Write-Host "[suite] DRYRUN -- no se ha lanzado ninguna sesion."
    return
}
Write-Host "[suite] Calendario registrado en $scheduleFile"

if (-not $NoAnalysis) {
    $pyExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
    $outDir = "out\$Suite"
    Write-Host "[suite] Analizando sesiones de esta tanda -> $outDir"
    & $pyExe "tools\analyze_ablation.py" "logs\sessions" "--suite" $Suite "--out" $outDir "--since-epoch" "$sinceEpoch"
}
