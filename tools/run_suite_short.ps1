<#
run_suite_short.ps1 - tanda corta de comprobacion (suite ablation_short).

Los mismos A1-A5, brazos SUB/ATOM y semilla que run_suite_long.ps1, pero con 2
repeticiones: 5 x 2 x 2 = 20 sesiones. Sirve para ver que todo funciona (A5
incluido) antes de lanzar la larga. Reutiliza run_suite.ps1 tal cual: intercalado
y contrabalanceo de brazos, arbol git limpio, build de Unity por manifiesto, check
de infraestructura, Unity en segundo plano y analisis al final en out\ablation_short.

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_suite_short.ps1 -DryRun      (solo calendario)
  powershell -ExecutionPolicy Bypass -File tools\run_suite_short.ps1              (20 sesiones)
  powershell -ExecutionPolicy Bypass -File tools\run_suite_short.ps1 -MaxHours 2  (no empieza repeticiones nuevas pasadas 2 h)
#>
param(
    [int]$Runs = 0,            # 0 = las 2 de la suite
    [double]$MaxHours = 0,     # 0 = sin limite
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoAnalysis,
    [switch]$ShowWindow
)

$ErrorActionPreference = "Stop"
$suiteArgs = @{ Suite = "ablation_short" }
if ($Runs -gt 0) { $suiteArgs["Runs"] = $Runs }
if ($MaxHours -gt 0) { $suiteArgs["MaxHours"] = $MaxHours }
if ($DryRun) { $suiteArgs["DryRun"] = $true }
if ($Force) { $suiteArgs["Force"] = $true }
if ($NoAnalysis) { $suiteArgs["NoAnalysis"] = $true }
if ($ShowWindow) { $suiteArgs["ShowWindow"] = $true }
& "$PSScriptRoot\run_suite.ps1" @suiteArgs
