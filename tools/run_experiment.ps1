<#
run_experiment.ps1 — Fase 13. Ejecuta N runs de un experimento (manifiesto
tools\experiments\<Id>.json) para una config dada (o todas las del
manifiesto), invocando test_session.ps1 por cada run y registrando cada uno
en logs\experiments\<Id>\runs.jsonl.

No duplica la lógica de arranque (Ollama/servidor/Unity) — eso lo hace
test_session.ps1; este script solo traduce el manifiesto+config a los flags
que test_session.ps1 ya entiende (-MemoryOff/-MemoryReuse/-MemoryDir,
-Npcs/-GoalsFile) más las variables de entorno que test_session.ps1 NO toca
(NPC_CANONICAL_REUSE/NPC_CANONICAL_FAMILY/NPC_COORDINATION,
NPC_EXPERIMENT_ID/NPC_CONFIG_LABEL/NPC_EXPERIMENT_N_OPT/NPC_EXPERIMENT_NPCS).

Pares M1→M2 (memoria episódica entre sesiones, Fase 6.5): se pre-crea un
directorio de run compartido y se pasa -MemoryDir a AMBAS sesiones del par —
M1 lo puebla (vacío al empezar → "reuse" == fresh + persistir), M2 lo reusa.
Batería memory_reuse: SUB_M1→SUB_M2 y ATOM_M1→ATOM_M2 son los brazos SUB/ATOM
con ese mismo mecanismo de pares (directorio por experimento, brazo y repetición).

Uso:
  powershell -ExecutionPolicy Bypass -File tools\run_experiment.ps1 -Id E1
  powershell -ExecutionPolicy Bypass -File tools\run_experiment.ps1 -Id E3 -Config M0 -Runs 3
  powershell -ExecutionPolicy Bypass -File tools\run_experiment.ps1 -Id E1 -DryRun
#>
param(
    [Parameter(Mandatory = $true)][string]$Id,
    [string]$Config = "",          # vacío = todas las configs del manifiesto
    [int]$Runs = 0,                # 0 = usa el N del manifiesto por config
    [switch]$DryRun,
    [switch]$Force,                # salta el check de arbol git limpio
    [int]$RunIndex = 0,            # Fase 16: indice de repeticion fijado por run_suite.ps1 (0 = contador local)
    [string]$SuiteId = "",         # Fase 16: suite a la que pertenece el run (solo registro)
    [switch]$ShowWindow,           # Unity con ventana (se pausa sin foco); por defecto segundo plano
    # OJO: Build_V1.2.0 es ANTERIOR a la Fase 11 y NO entiende el formato
    # goals.json por-NPC ({"npcs":[...]})  -- con ese build, LoadGoalsOverride
    # falla en silencio (JSON sin "goals" en la raíz) y cae al goal por
    # defecto del inspector, dando resultados sin sentido (confirmado en el
    # piloto de E1, 2026-08-11). Build_Fase13_test (2026-08-11) añade el
    # item craft-only 'flour' + NpcSpawnSpec (npc_miller/npc_baker) sobre el
    # de la Fase 11 -- necesario para E5/E6.
    [string]$BuildExe = "builds\coop\My project.exe"
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

function Get-GitSha {
    try { $sha = (git rev-parse --short HEAD 2>$null); if ($LASTEXITCODE -eq 0) { return $sha } } catch {}
    return ""
}

function Assert-CleanTree {
    if ($Force) {
        Write-Host "[exp] AVISO: -Force -- se ignora el estado del arbol git (los resultados NO son reproducibles por commit)."
        return
    }
    # Solo bloquea por cambios en ficheros YA TRACKEADOS (modificados/staged) --
    # eso es lo que determina si el código en el commit == código que corrió.
    # Ficheros sueltos sin trackear ('??', p.ej. notas del autor) no afectan
    # la reproducibilidad del código versionado y no deben bloquear la tanda.
    $status = git status --porcelain 2>$null | Where-Object { $_ -notmatch '^\?\? ' }
    if ($status) {
        throw "Arbol de git sucio -- congela el commit antes de correr experimentos (regla de reproducibilidad, ver PROTOCOLO_EXPERIMENTOS.md sec. 0), o usa -Force si es intencional. Cambios:`n$status"
    }
}

function ConvertTo-UnityGoalsFile {
    param($GoalsObj, [string]$OutPath)
    # $GoalsObj: PSCustomObject npc_id -> [ {nl,condition}, ... ] (del manifiesto,
    # comodo de escribir a mano). Se traduce a la LISTA npcs:[{npc_id,goals}]
    # que exige JsonUtility en Unity -- JsonUtility no deserializa
    # Dictionary<string,T> (ver D1 de FASE_11_MULTIAGENTE_INDEPENDIENTE.md).
    $npcsList = @()
    foreach ($npcId in $GoalsObj.PSObject.Properties.Name) {
        $npcsList += [PSCustomObject]@{ npc_id = $npcId; goals = $GoalsObj.$npcId }
    }
    $doc = [PSCustomObject]@{ npcs = $npcsList }
    $doc | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 $OutPath
}

function Set-ConfigEnv {
    param([string]$Cfg)
    Remove-Item Env:NPC_CANONICAL_REUSE, Env:NPC_CANONICAL_FAMILY, Env:NPC_COORDINATION, `
        Env:NPC_GOAL_ARBITRATION, Env:NPC_ARBITRATION_MODE, Env:NPC_BUILTIN_SUBPLANS, `
        Env:NPC_COORDINATION_PLANNER -ErrorAction SilentlyContinue
    switch ($Cfg) {
        "M0" { }
        # Fase 16 - ablacion de sub-planes: MISMA config base (LLM sin memoria, sin
        # familia, sin coordinacion). UNICA diferencia entre brazos: NPC_BUILTIN_SUBPLANS.
        # Fase 17: con coordinacion (manifiestos CO*), SUB/ATOM planifican con el LLM.
        "SUB"  { $env:NPC_BUILTIN_SUBPLANS = "1"; $env:NPC_COORDINATION_PLANNER = "llm" }
        "ATOM" { $env:NPC_BUILTIN_SUBPLANS = "0"; $env:NPC_COORDINATION_PLANNER = "llm" }
        # Fase 17 - referencia determinista para coordinacion: familia de planes + plan
        # de entrega (sin LLM de planificacion), con sub-planes.
        "DET" {
            $env:NPC_BUILTIN_SUBPLANS = "1"; $env:NPC_CANONICAL_REUSE = "1"; $env:NPC_CANONICAL_FAMILY = "1"
            $env:NPC_COORDINATION_PLANNER = "family"
        }
        # Bateria memory_reuse: brazos SUB/ATOM con memoria de planes en pares
        # (1.a sesion aprende, 2.a reusa). Sin clave canonica: el sig de parse_goals
        # es estable en A4/A5 (32/32 en la tanda larga) y SUB/ATOM quedan intactos.
        "SUB_M1"  { $env:NPC_BUILTIN_SUBPLANS = "1"; $env:NPC_COORDINATION_PLANNER = "llm" }
        "SUB_M2"  { $env:NPC_BUILTIN_SUBPLANS = "1"; $env:NPC_COORDINATION_PLANNER = "llm" }
        "ATOM_M1" { $env:NPC_BUILTIN_SUBPLANS = "0"; $env:NPC_COORDINATION_PLANNER = "llm" }
        "ATOM_M2" { $env:NPC_BUILTIN_SUBPLANS = "0"; $env:NPC_COORDINATION_PLANNER = "llm" }
        "M1" { $env:NPC_CANONICAL_REUSE = "1" }
        "M2" { $env:NPC_CANONICAL_REUSE = "1" }
        "C1" { $env:NPC_CANONICAL_REUSE = "1"; $env:NPC_CANONICAL_FAMILY = "1" }
        # Fase 14: X1 lleva arbitraje de goals vía LLM (con fallback a la regla
        # determinista si falla/timeout/no valida); X2 es el control -- misma
        # preempción, pero SIEMPRE por regla, sin LLM. Para E1-E5 el arbitraje
        # activado es un no-op (nunca hay un segundo goal candidato mientras la
        # intención activa espera a un peer), así que X1 sigue siendo comparable
        # a las sesiones de E5 pre-Fase-14 (ver EVALUACION_RESULTADOS.md §4).
        "X1" {
            $env:NPC_CANONICAL_REUSE = "1"; $env:NPC_CANONICAL_FAMILY = "1"; $env:NPC_COORDINATION = "1"
            $env:NPC_GOAL_ARBITRATION = "1"; $env:NPC_ARBITRATION_MODE = "llm"
        }
        "X2" {
            $env:NPC_CANONICAL_REUSE = "1"; $env:NPC_CANONICAL_FAMILY = "1"; $env:NPC_COORDINATION = "1"
            $env:NPC_GOAL_ARBITRATION = "1"; $env:NPC_ARBITRATION_MODE = "rule"
        }
        default { throw "Config desconocida: '$Cfg' (ver tools\experiments\configs.json)" }
    }
}

Assert-CleanTree
$gitSha = Get-GitSha
Write-Host "[exp] commit: $gitSha"

# Ninguno de E1-E6 mide reactividad (Fase 9) -- el trigger demostrativo de
# inventory.asl (wheat>=2 -> adopta achieve_bake_bread) contamina experimentos
# de un solo goal aislado con un segundo goal no pedido (confirmado en el
# piloto de E2, 2026-08-11: goals_started se duplicaba). Desactivado para
# TODA la batería.
$env:NPC_BUILTIN_TRIGGERS = "0"
# Fase 16: contratos de capacidad aislados por sesion (sin contaminacion entre runs;
# ademas evita modificar src/plans/contracts/, que esta versionado).
$env:NPC_ISOLATE_CONTRACTS = "1"

$manifestPath = "tools\experiments\$Id.json"
if (-not (Test-Path $manifestPath)) { throw "No existe el manifiesto: $manifestPath" }
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Build de Unity por manifiesto (build_exe): los escenarios de un NPC (E1-E4, A1-A4)
# necesitan builds\single (NPC por defecto npc_001); los de coordinacion,
# builds\coop (NpcSpawnSpec npc_miller/npc_baker). Un -BuildExe explicito manda.
if (-not $PSBoundParameters.ContainsKey('BuildExe') -and $manifest.build_exe) {
    $BuildExe = [string]$manifest.build_exe
}
Write-Host "[exp] build: $BuildExe"
if (-not $DryRun -and -not (Test-Path $BuildExe)) { throw "No existe la build: $BuildExe" }

$configsToRun = if ($Config) { @($Config) } else { $manifest.configs }

$expLogDir = "logs\experiments\$Id"
New-Item -ItemType Directory -Force $expLogDir | Out-Null
$runsJsonl = Join-Path $expLogDir "runs.jsonl"

$goalsTempDir = Join-Path $env:TEMP "npc_experiments"
New-Item -ItemType Directory -Force $goalsTempDir -ErrorAction SilentlyContinue | Out-Null
$goalsFile = Join-Path $goalsTempDir "$Id`_goals.json"
ConvertTo-UnityGoalsFile -GoalsObj $manifest.goals -OutPath $goalsFile
$manifestEnvKeys = @()
Write-Host "[exp] goals.json generado: $goalsFile"

try {
    foreach ($cfg in $configsToRun) {
        $nRuns = if ($Runs -gt 0) { $Runs } elseif ($manifest.runs.PSObject.Properties.Name -contains $cfg) { $manifest.runs.$cfg } else { 0 }
        if (-not $nRuns) {
            Write-Host "[exp] Config '$cfg' sin N definido en el manifiesto -- saltando"
            continue
        }

        Write-Host "[exp] === $Id / $cfg -- $nRuns runs ==="
        $isPaired = ($cfg -eq "M1" -or $cfg -eq "M2" -or $cfg -like "*_M1" -or $cfg -like "*_M2")

        for ($i = 1; $i -le $nRuns; $i++) {
            $env:NPC_EXPERIMENT_ID = $Id
            $env:NPC_CONFIG_LABEL = $cfg
            if ($manifest.n_opt) { $env:NPC_EXPERIMENT_N_OPT = "$($manifest.n_opt)" }
            if ($manifest.npcs) { $env:NPC_EXPERIMENT_NPCS = "$($manifest.npcs)" }
            Set-ConfigEnv $cfg
            # Fase 16: cierre ORDENADO en Python antes del kill del lanzador
            # (metrics.json escrito y goals abiertos registrados como timeout).
            $env:NPC_SESSION_MAX_S = "$([math]::Max(30, [int]$manifest.timeout_s - 25))"
            # Fase 17: variables de ESCENARIO del manifiesto (p.ej. coordinacion y arbitraje
            # en CO5/CO6), aplicadas despues de la config: la config fija el brazo y el
            # manifiesto el escenario.
            if ($manifest.env) {
                foreach ($prop in $manifest.env.PSObject.Properties) {
                    Set-Item -Path "Env:$($prop.Name)" -Value "$($prop.Value)"
                    $manifestEnvKeys += $prop.Name
                }
            }

            $sessionArgs = @{
                TimeoutSec = $manifest.timeout_s
                BuildExe   = $BuildExe
                GoalsFile  = $goalsFile
            }
            if ($manifest.npcs -and $manifest.npcs -gt 1) { $sessionArgs["Npcs"] = $manifest.npcs }
            if ($ShowWindow) { $sessionArgs["ShowWindow"] = $true }

            if ($cfg -eq "M0" -or $cfg -eq "SUB" -or $cfg -eq "ATOM" -or $cfg -eq "DET") {
                $sessionArgs["MemoryOff"] = $true
            } elseif ($isPaired) {
                $pairDir = Join-Path $proj "plans\memory\runs\exp_${Id}_pair$i"
                if ($cfg -like "*_M1" -or $cfg -like "*_M2") {
                    # Bateria memory_reuse: un directorio por experimento, brazo y
                    # repeticion (run_suite_memory.ps1 fija -RunIndex; con -Runs 1 el
                    # contador local $i seria siempre 1 y los pares se pisarian).
                    $arm = $cfg.Substring(0, $cfg.LastIndexOf("_"))
                    $pairIdx = if ($RunIndex -gt 0) { $RunIndex } else { $i }
                    $pairDir = Join-Path $proj "plans\memory\runs\mem_${Id}_${arm}_rep$pairIdx"
                }
                if ($cfg -eq "M1" -or $cfg -like "*_M1") {
                    if (Test-Path $pairDir) { Remove-Item -Recurse -Force $pairDir }
                    New-Item -ItemType Directory -Force $pairDir | Out-Null
                    Write-Host "[exp] par M1->M2 #$i -- run_dir: $pairDir"
                }
                $sessionArgs["MemoryDir"] = $pairDir
            }
            # C1/X1: memoria on, run nuevo cada vez (sin flags extra de memoria).

            Write-Host "[exp] --- $Id/$cfg run $i/$nRuns ---"
            if ($DryRun) {
                Write-Host "[exp] DRYRUN args: $($sessionArgs | ConvertTo-Json -Compress)"
                continue
            }

            $t0 = Get-Date
            $exitOk = $true
            try {
                & "$PSScriptRoot\test_session.ps1" @sessionArgs
            } catch {
                $exitOk = $false
                Write-Host "[exp] AVISO: run fallo: $($_.Exception.Message)"
            }
            $elapsed = ((Get-Date) - $t0).TotalSeconds

            $sessionSubdir = $null
            $latestSession = Get-ChildItem "logs\sessions" -Directory -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestSession) {
                $sub = Get-ChildItem $latestSession.FullName -Directory -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($sub) { $sessionSubdir = $sub.FullName }
            }

            $record = [PSCustomObject]@{
                experiment_id = $Id
                config_label  = $cfg
                run_index     = $(if ($RunIndex -gt 0) { $RunIndex } else { $i })
                suite_id      = $SuiteId
                git_sha       = $gitSha
                session_dir   = $sessionSubdir
                elapsed_s     = [math]::Round($elapsed, 1)
                ok            = $exitOk
                started_at    = $t0.ToString("o")
                unity_mode    = $(if ($ShowWindow) { "window" } else { "batchmode" })
            }
            Add-Content -Path $runsJsonl -Value ($record | ConvertTo-Json -Compress)

            # Comprobacion de infraestructura: cada NPC con goal en el manifiesto tiene que
            # haberse registrado desde Unity. Si no (build o spawn equivocados), la sesion no
            # mide nada: se aborta la tanda en vez de gastar horas.
            $traceFile = if ($sessionSubdir) { Join-Path $sessionSubdir "trace.jsonl" } else { $null }
            if ($traceFile -and (Test-Path $traceFile)) {
                $traceText = Get-Content $traceFile -Raw
                $missing = @()
                foreach ($npcId in $manifest.goals.PSObject.Properties.Name) {
                    $pattern = '"msg_type": "RegisterNPC", "payload": \{"npc_id": "' + [regex]::Escape($npcId) + '"'
                    if ($traceText -notmatch $pattern) { $missing += $npcId }
                }
                if ($missing.Count -gt 0) {
                    throw "INFRA: el NPC con goal ($($missing -join ', ')) no se registro desde Unity en $sessionSubdir. Revisa la build ($BuildExe) y el spawn de NPCs. Tanda abortada."
                }
            }
        }
    }
} finally {
    Remove-Item Env:NPC_EXPERIMENT_ID, Env:NPC_CONFIG_LABEL, Env:NPC_EXPERIMENT_N_OPT, Env:NPC_EXPERIMENT_NPCS, `
        Env:NPC_CANONICAL_REUSE, Env:NPC_CANONICAL_FAMILY, Env:NPC_COORDINATION, Env:NPC_BUILTIN_TRIGGERS, `
        Env:NPC_GOAL_ARBITRATION, Env:NPC_ARBITRATION_MODE, Env:NPC_BUILTIN_SUBPLANS, `
        Env:NPC_SESSION_MAX_S, Env:NPC_ISOLATE_CONTRACTS, Env:NPC_COORDINATION_PLANNER `
        -ErrorAction SilentlyContinue
    foreach ($k in $manifestEnvKeys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}

Write-Host "[exp] Listo. Runs registrados en $runsJsonl"
