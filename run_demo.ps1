<#
run_demo.ps1 — Ejecuta una demo de demos\ con el sistema completo y Unity en ventana.

  1. Comprueba Ollama y el modelo configurado en src\config\settings.json.
  2. Crea el entorno virtual e instala dependencias si no existen.
  3. Arranca el servidor Python (XMPP embebido + TCP 7777 + Agente Planificador).
  4. Abre la build de Unity con los objetivos de la demo.

Al cerrar la ventana de Unity se detiene también el servidor.

Uso:
  run_demo.cmd -List                    # lista las demos disponibles
  run_demo.cmd -Demo A3                 # un NPC fabrica pan (por defecto)
  run_demo.cmd -Demo CO6                # molinero y panadero se ayudan en ambos sentidos
  run_demo.cmd -Demo A3 -NoSubplans     # sin planes auxiliares (configuración ATOM)
  run_demo.cmd -Demo CO5 -Deterministic # planes deterministas (configuración DET)
  run_demo.cmd -GoalsFile mis_objetivos.json -Build single

Parámetros:
  -Demo <nombre>          Fichero demos\<nombre>.json. Por defecto: A3.
  -List                   Muestra las demos y termina.
  -NoSubplans             El modelo compone solo acciones primitivas (ATOM).
  -Deterministic          Planes generados sin el modelo (DET, demos de dos NPC).
  -GoalsFile <ruta>       Objetivos propios en el formato de goals.json, en lugar de una demo.
  -Build single|coop      Build para -GoalsFile. Por defecto: single.
  -Model <nombre>         Modelo de Ollama (por defecto, el de settings.json).
#>
param(
    [string]$Demo = "A3",
    [switch]$List,
    [switch]$NoSubplans,
    [switch]$Deterministic,
    [string]$GoalsFile = "",
    [ValidateSet("single", "coop")]
    [string]$Build = "single",
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Demos disponibles ─────────────────────────────────────────────────────────
if ($List) {
    Get-ChildItem "demos\*.json" | ForEach-Object {
        $d = Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        "{0,-18} {1}" -f $_.BaseName, $d.title
        "{0,-18} {1}" -f "", $d.description
    }
    exit 0
}

if ($NoSubplans -and $Deterministic) { throw "-NoSubplans y -Deterministic no se pueden combinar." }

# ── Demo u objetivos propios ──────────────────────────────────────────────────
New-Item -ItemType Directory -Force "logs" | Out-Null
$demoEnv = @{}
$memory = "off"
$memoryGroup = ""
$npcs = 0

if ($GoalsFile) {
    if (-not (Test-Path $GoalsFile)) { throw "No existe el fichero de objetivos: $GoalsFile" }
    $buildName = $Build
    $goalsPath = (Resolve-Path $GoalsFile).Path
    $label = $GoalsFile
} else {
    $demoPath = "demos\$Demo.json"
    if (-not (Test-Path $demoPath)) { throw "No existe la demo '$Demo'. Usa -List para ver las disponibles." }
    $d = Get-Content $demoPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $buildName = $d.build
    $npcs = [int]$d.npcs
    $label = "$Demo · $($d.title)"
    if ($d.env) { foreach ($p in $d.env.PSObject.Properties) { $demoEnv[$p.Name] = [string]$p.Value } }
    if ($d.memory) { $memory = $d.memory }
    if ($d.memory_group) { $memoryGroup = $d.memory_group }

    # Unity lee una lista npcs:[{npc_id, goals}] (JsonUtility no admite diccionarios).
    $npcList = @()
    foreach ($p in $d.goals.PSObject.Properties) {
        $npcList += [PSCustomObject]@{ npc_id = $p.Name; goals = @($p.Value) }
    }
    $goalsPath = Join-Path $PSScriptRoot "logs\demo_goals_$Demo.json"
    [PSCustomObject]@{ npcs = $npcList } | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $goalsPath
}

if ($Deterministic -and $buildName -ne "coop") { throw "-Deterministic solo está disponible en las demos de dos NPC." }

$buildExe = "builds\$buildName\My project.exe"
if (-not (Test-Path $buildExe)) {
    throw "No se encuentra '$buildExe'. Descarga builds.zip de la última release y descomprímelo en la raíz del repositorio (deben quedar builds\single y builds\coop)."
}
$buildExe = (Resolve-Path $buildExe).Path

# ── 1. Ollama y modelo ────────────────────────────────────────────────────────
$settings = Get-Content "src\config\settings.json" -Raw | ConvertFrom-Json
$model = if ($Model) { $Model } else { $settings.llm_model }
$ollamaRoot = ($settings.llm_base_url -replace '/v1/?$', '').TrimEnd('/')

function Get-OllamaTags {
    try { return Invoke-RestMethod -TimeoutSec 3 "$ollamaRoot/api/tags" } catch { return $null }
}

$tags = Get-OllamaTags
if (-not $tags) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama no está instalado o no está en el PATH. Instálalo desde https://ollama.com y ejecuta: ollama pull $model"
    }
    Write-Host "[demo] Arrancando Ollama..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 20 -and -not $tags; $i++) { Start-Sleep -Seconds 1; $tags = Get-OllamaTags }
    if (-not $tags) { throw "Ollama no responde en $ollamaRoot" }
}
$installed = @($tags.models | ForEach-Object { $_.name })
if ($installed -notcontains $model -and $installed -notcontains "$model`:latest") {
    throw "El modelo '$model' no está descargado. Ejecuta: ollama pull $model"
}
Write-Host "[demo] Ollama OK, modelo $model"

# ── 2. Entorno de Python ──────────────────────────────────────────────────────
$pyExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pyExe)) {
    Write-Host "[demo] Creando entorno virtual (.venv) con Python 3.12..."
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv } else { python -m venv .venv }
    if (-not (Test-Path $pyExe)) { throw "No se pudo crear .venv. Instala Python 3.12 (spade-llm requiere >=3.11,<3.13)." }
    & $pyExe -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Falló la instalación de dependencias." }
}

# ── 3. Configuración de la sesión ─────────────────────────────────────────────
foreach ($name in @(
        "NPC_BUILTIN_SUBPLANS", "NPC_COORDINATION_PLANNER", "NPC_CANONICAL_REUSE", "NPC_CANONICAL_FAMILY",
        "NPC_COORDINATION", "NPC_GOAL_ARBITRATION", "NPC_ARBITRATION_MODE",
        "NPC_PLAN_MEMORY_ENABLED", "NPC_PLAN_MEMORY_REUSE", "NPC_PLAN_MEMORY_DIR")) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:NPC_LLM_MODEL_OVERRIDE = $model
$env:NPC_BUILTIN_TRIGGERS = "0"      # sin objetivos reactivos de ejemplo
$env:NPC_ISOLATE_CONTRACTS = "1"     # no modifica src\plans\contracts
$env:NPC_COORDINATION_PLANNER = "llm"
$env:NPC_BUILTIN_SUBPLANS = if ($NoSubplans) { "0" } else { "1" }
if ($Deterministic) {
    $env:NPC_CANONICAL_REUSE = "1"
    $env:NPC_CANONICAL_FAMILY = "1"
    $env:NPC_COORDINATION_PLANNER = "family"
}
foreach ($key in $demoEnv.Keys) { Set-Item -Path "Env:$key" -Value $demoEnv[$key] }

switch ($memory) {
    "off" { $env:NPC_PLAN_MEMORY_ENABLED = "0" }
    "learn" {
        $memDir = Join-Path $PSScriptRoot "plans\memory\runs\demo_$memoryGroup"
        if (Test-Path $memDir) { Remove-Item -Recurse -Force $memDir }
        New-Item -ItemType Directory -Force $memDir | Out-Null
        $env:NPC_PLAN_MEMORY_ENABLED = "1"
        $env:NPC_PLAN_MEMORY_DIR = $memDir
        Write-Host "[demo] Memoria de planes: se guardará en $memDir"
    }
    "reuse" {
        $memDir = Join-Path $PSScriptRoot "plans\memory\runs\demo_$memoryGroup"
        if (-not (Test-Path $memDir) -or -not (Get-ChildItem $memDir -Recurse -File)) {
            throw "No hay memoria guardada en $memDir. Ejecuta antes la demo de aprendizaje (MEM-1-aprender) hasta que resuelva sus objetivos."
        }
        $env:NPC_PLAN_MEMORY_ENABLED = "1"
        $env:NPC_PLAN_MEMORY_DIR = $memDir
        Write-Host "[demo] Memoria de planes: se reutiliza $memDir"
    }
    default { throw "Valor de memory no válido en la demo: $memory" }
}

# ── 4. Servidor ───────────────────────────────────────────────────────────────
Write-Host "[demo] $label"
Write-Host "[demo] Arrancando el servidor (se abre en otra ventana)..."
$pyProc = Start-Process -FilePath (Resolve-Path $pyExe).Path -ArgumentList "src\main.py" -PassThru

$port = [int]$settings.unity_port
$ready = $false
for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
    Start-Sleep -Seconds 1
    if ($pyProc.HasExited) { throw "El servidor se ha cerrado al arrancar. Revisa su ventana o logs\." }
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $port)
        $client.Close()
        $ready = $true
    } catch { }
}
if (-not $ready) { Stop-Process -Id $pyProc.Id -Force; throw "El servidor no abrió el puerto $port en 60 s." }
Write-Host "[demo] Servidor escuchando en 127.0.0.1:$port"

# ── 5. Unity ──────────────────────────────────────────────────────────────────
$unityArgs = @("-goalsFile", "`"$goalsPath`"")
if ($npcs -gt 1) { $unityArgs += @("-npcs", "$npcs") }
Write-Host "[demo] Abriendo Unity. Mantén su ventana en primer plano: la simulación se pausa si pierde el foco."
try {
    $unityProc = Start-Process -FilePath $buildExe -ArgumentList $unityArgs -PassThru
    $unityProc.WaitForExit()
} finally {
    if (-not $pyProc.HasExited) { Stop-Process -Id $pyProc.Id -Force }
    Write-Host "[demo] Sesión terminada. Trazas en logs\sessions\"
}
