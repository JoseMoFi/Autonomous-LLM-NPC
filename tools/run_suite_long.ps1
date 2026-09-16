<#
run_suite_long.ps1 - tanda larga en DOS FASES (ablacion + cooperacion).

  Fase 1  suite ablation_long        A1-A5 x SUB/ATOM x 16 = 160 sesiones
          analisis -> out\ablation_long
  Fase 2  suite coordination_builtins CO5/CO6 x SUB/ATOM/DET x 6 = 36 sesiones
          analisis -> out\coordination_builtins

Las dos fases son independientes: si una aborta (arbol sucio, INFRA, etc.) la
otra se lanza igual y al final se resume que paso en cada una. Cada fase usa
run_suite.ps1 tal cual (intercalado y contrabalanceo de brazos, build por
manifiesto, check de infraestructura, Unity en segundo plano y analisis propio).

-MaxHours se reparte entre las dos fases: la segunda recibe lo que quede. Si no
queda tiempo, se omite avisando. El corte solo ocurre al empezar una repeticion,
asi que las completadas siguen contrabalanceadas.

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_suite_long.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File tools\run_suite_long.ps1                  (196 sesiones)
  powershell -ExecutionPolicy Bypass -File tools\run_suite_long.ps1 -MaxHours 8      (toda la noche, con tope)
  powershell -ExecutionPolicy Bypass -File tools\run_suite_long.ps1 -Phase coop -CoopRuns 1   (piloto de cooperacion)
#>
param(
    [int]$Runs = 0,          # 0 = las 16 repeticiones de ablation_long
    [int]$CoopRuns = 0,      # 0 = las 6 repeticiones de coordination_builtins
    [double]$MaxHours = 0,   # 0 = sin limite; se reparte entre las dos fases
    [ValidateSet("all", "ablation", "coop")][string]$Phase = "all",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoAnalysis,
    [switch]$ShowWindow
)

$ErrorActionPreference = "Stop"
$t0 = Get-Date
$results = @()

function Invoke-SuitePhase {
    param([string]$Label, [string]$Suite, [int]$PhaseRuns)

    $remaining = 0.0
    if ($MaxHours -gt 0) {
        $remaining = $MaxHours - ((Get-Date) - $t0).TotalHours
        if ($remaining -le 0.05) {
            Write-Host ""
            Write-Host ("[long] {0} OMITIDA: agotado el tope de {1} h." -f $Label, $MaxHours)
            return [PSCustomObject]@{ fase = $Label; estado = "omitida (MaxHours)" }
        }
    }

    $suiteArgs = @{ Suite = $Suite }
    if ($PhaseRuns -gt 0) { $suiteArgs["Runs"] = $PhaseRuns }
    if ($remaining -gt 0) { $suiteArgs["MaxHours"] = [math]::Round($remaining, 3) }
    if ($DryRun) { $suiteArgs["DryRun"] = $true }
    if ($Force) { $suiteArgs["Force"] = $true }
    if ($NoAnalysis) { $suiteArgs["NoAnalysis"] = $true }
    if ($ShowWindow) { $suiteArgs["ShowWindow"] = $true }

    Write-Host ""
    Write-Host ("================ {0} ({1}) ================" -f $Label, $Suite)
    try {
        & "$PSScriptRoot\run_suite.ps1" @suiteArgs
        return [PSCustomObject]@{ fase = $Label; estado = "ok" }
    } catch {
        Write-Host ("[long] {0} FALLO: {1}" -f $Label, $_.Exception.Message)
        return [PSCustomObject]@{ fase = $Label; estado = ("fallo: " + $_.Exception.Message) }
    }
}

if ($Phase -ne "coop") {
    $results += Invoke-SuitePhase -Label "Fase 1 - ablacion" -Suite "ablation_long" -PhaseRuns $Runs
}
if ($Phase -ne "ablation") {
    $results += Invoke-SuitePhase -Label "Fase 2 - cooperacion" -Suite "coordination_builtins" -PhaseRuns $CoopRuns
}

$totalMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host ""
Write-Host "================ Resumen ================"
foreach ($r in $results) { Write-Host ("[long] {0}: {1}" -f $r.fase, $r.estado) }
Write-Host "[long] Tiempo total: $totalMin min."
if (-not $DryRun -and -not $NoAnalysis) {
    Write-Host "[long] Resultados:"
    Write-Host "[long]   out\ablation_long\ABLACION_SUBPLANES_RESULTADOS.md"
    Write-Host "[long]   out\coordination_builtins\ABLACION_SUBPLANES_RESULTADOS.md"
}
if ($results | Where-Object { $_.estado -like "fallo*" }) { exit 1 }
