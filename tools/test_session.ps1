<#
test_session.ps1 — Lanzador de sesión end-to-end AUTÓNOMO para pruebas.

Arranca, en orden:
  1. Ollama (si no está corriendo) + warm-up del modelo de settings.json.
  2. El servidor Python (src/main.py): XMPP embebido + TCP 7777 + LLMPlanningAgent.
  3. La build de Unity (My project.exe).
Espera -TimeoutSec segundos (default 300 = 5 min) y luego MATA todos los
procesos que arrancó. Pensado para que un agente pueda probar sin intervención.

Uso:
  powershell -ExecutionPolicy Bypass -File tools\test_session.ps1
  powershell -ExecutionPolicy Bypass -File tools\test_session.ps1 -TimeoutSec 180
  powershell -ExecutionPolicy Bypass -File tools\test_session.ps1 -BuildExe "builds\coop\My project.exe" -Npcs 2 -GoalsFile mis_objetivos.json

Al terminar imprime el directorio de la sesión (logs/sessions/<fecha>/<hora>/)
para inspeccionar trace.jsonl.
#>
param(
    [int]$TimeoutSec = 300,
    # Build de Unity de la release: builds\single (un NPC) o builds\coop
    # (npc_miller y npc_baker). Ruta relativa a la raiz del repositorio.
    [string]$BuildExe = "builds\single\My project.exe",
    # Plan memory (Fase 4): -MemoryReuse reutiliza un run; -MemoryDir <ruta> uno
    # concreto; -MemoryOff lo desactiva. Por defecto: run nuevo cada sesión.
    [switch]$MemoryReuse,
    [string]$MemoryDir = "",
    [switch]$MemoryOff,
    # Matriz de experimentos (Fase 5): -Model <ollama> sobreescribe el modelo;
    # -NoRefinement desactiva el refinamiento V4 (config B).
    [string]$Model = "",
    [switch]$NoRefinement,
    # Fase 11 (T8): -Npcs N sobreescribe NPCFactory.spawnCount (0 = sin
    # override, usa el del inspector); -GoalsFile <ruta> sobreescribe el JSON
    # de goals que lee NpcProfileSender (vacío = ruta por defecto del repo).
    # Ambos se pasan tal cual como argumentos de línea de comandos al build.
    [int]$Npcs = 0,
    [string]$GoalsFile = "",
    # Por defecto la build corre en segundo plano (-batchmode -nographics): sin
    # ventana y sin pausarse al perder el foco (la build tiene runInBackground=0,
    # asi que con ventana se congela si el usuario cambia de aplicacion). La
    # simulacion no depende del render. -ShowWindow = comportamiento anterior.
    [switch]$ShowWindow
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

$startedOllama = $false
$pyProc = $null
$unityProc = $null

function Resolve-Settings {
    $s = Get-Content "src\config\settings.json" -Raw | ConvertFrom-Json
    return $s
}

try {
    $settings = Resolve-Settings
    $model = if ($Model) { $Model } else { $settings.llm_model }
    $ollamaRoot = ($settings.llm_base_url -replace '/v1/?$','').TrimEnd('/')

    # ── 1. Ollama ────────────────────────────────────────────────────────────
    $running = $false
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "$ollamaRoot/api/tags" | Out-Null
        $running = $true
    } catch { $running = $false }

    if (-not $running) {
        Write-Host "[test] Arrancando 'ollama serve'..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        $startedOllama = $true
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try {
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "$ollamaRoot/api/tags" | Out-Null
                $running = $true; break
            } catch {}
        }
    }
    if (-not $running) { throw "Ollama no respondió en $ollamaRoot" }
    Write-Host "[test] Ollama OK en $ollamaRoot"

    Write-Host "[test] Warm-up del modelo '$model'..."
    $body = @{ model = $model; prompt = ""; stream = $false; keep_alive = "30m" } | ConvertTo-Json
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 120 -Method Post `
            -ContentType "application/json" -Body $body "$ollamaRoot/api/generate" | Out-Null
        Write-Host "[test] Modelo cargado."
    } catch {
        Write-Host "[test] AVISO: warm-up no completado: $($_.Exception.Message)"
    }

    # ── 2. Servidor Python ────────────────────────────────────────────────────
    if (-not (Test-Path "logs")) { New-Item -ItemType Directory logs | Out-Null }
    $serverLog = "logs\test_session_server.log"
    Write-Host "[test] Arrancando servidor Python (src/main.py) → $serverLog ..."
    $env:NPC_LLM_MODEL_OVERRIDE = $model
    # Apagado por inactividad: el servidor se cierra solo cuando todos los goals
    # se resuelven (completados o fallados sin repair), sin esperar el timeout.
    $env:NPC_SHUTDOWN_WHEN_IDLE = "1"
    # Plan memory: por defecto run nuevo; flags para reuse/dir/off.
    Remove-Item Env:NPC_PLAN_MEMORY_ENABLED, Env:NPC_PLAN_MEMORY_REUSE, Env:NPC_PLAN_MEMORY_DIR -ErrorAction SilentlyContinue
    if ($MemoryOff) { $env:NPC_PLAN_MEMORY_ENABLED = "0" }
    if ($MemoryReuse) { $env:NPC_PLAN_MEMORY_REUSE = "1" }
    if ($MemoryDir) { $env:NPC_PLAN_MEMORY_DIR = $MemoryDir }
    # Refinamiento (config B de la matriz).
    Remove-Item Env:NPC_USE_REFINEMENT -ErrorAction SilentlyContinue
    if ($NoRefinement) { $env:NPC_USE_REFINEMENT = "0" }
    $pyExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
    $pyProc = Start-Process -FilePath $pyExe -ArgumentList "src\main.py" `
        -RedirectStandardOutput $serverLog -RedirectStandardError "logs\test_session_server.err.log" `
        -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 8  # dar tiempo a XMPP embebido + TCP 7777

    # ── 3. Build de Unity ─────────────────────────────────────────────────────
    if (-not (Test-Path $BuildExe)) { throw "No existe la build: $BuildExe" }
    $BuildExe = (Resolve-Path $BuildExe).Path
    if ($GoalsFile) { $GoalsFile = (Resolve-Path $GoalsFile).Path }
    $unityArgs = @()
    if (-not $ShowWindow) { $unityArgs += @("-batchmode", "-nographics") }
    Write-Host "[test] Modo Unity: $(if ($ShowWindow) { 'ventana (se pausa sin foco)' } else { 'segundo plano (batchmode, sin ventana)' })"
    if ($Npcs -gt 0) { $unityArgs += @("-npcs", "$Npcs") }
    if ($GoalsFile) { $unityArgs += @("-goalsFile", $GoalsFile) }
    if ($unityArgs.Count -gt 0) {
        Write-Host "[test] Arrancando build de Unity: $BuildExe $($unityArgs -join ' ')"
        $unityProc = Start-Process -FilePath $BuildExe -ArgumentList $unityArgs -PassThru
    } else {
        Write-Host "[test] Arrancando build de Unity: $BuildExe"
        $unityProc = Start-Process -FilePath $BuildExe -PassThru
    }

    # ── 4. Esperar (corta si el servidor se apaga solo por inactividad) ────────
    Write-Host "[test] Sesión en marcha. Máximo $TimeoutSec s (corta antes si se resuelven los goals)..."
    $elapsed = 0
    while ($elapsed -lt $TimeoutSec) {
        Start-Sleep -Seconds 3
        $elapsed += 3
        if ($pyProc.HasExited) {
            Write-Host "[test] El servidor se apagó solo (goals resueltos) tras ~$elapsed s."
            break
        }
    }
}
finally {
    Write-Host "[test] Cerrando procesos..."
    if ($unityProc -and -not $unityProc.HasExited) {
        Stop-Process -Id $unityProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[test] Unity detenido."
    }
    # Matar cualquier resto de la build por nombre (por si lanzó subprocesos).
    Get-Process "My project" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    if ($pyProc -and -not $pyProc.HasExited) {
        Stop-Process -Id $pyProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[test] Servidor Python detenido."
    }
    if ($startedOllama) {
        Get-Process ollama* -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Host "[test] Ollama detenido (lo arrancamos nosotros)."
    }

    # Imprimir la sesión más reciente para inspección.
    $latest = Get-ChildItem "logs\sessions" -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        $sub = Get-ChildItem $latest.FullName -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($sub) { Write-Host "[test] Sesión: $($sub.FullName)" }
    }
}
