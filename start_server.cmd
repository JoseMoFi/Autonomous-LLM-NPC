@echo off
:: ============================================================================
:: start_server.cmd  —  Lanzador V2  (Python BDI + LLM NPC para Unity)
:: Ubicar en:  raiz del repositorio
::
:: Uso:  start_server.cmd [-c] [-m <modelo>]
::
::   -c / --clear          Archiva logs de sesiones anteriores (inicio limpio)
::   -m / --model <name>   Sobreescribe llm_model de settings.json
::
:: Configuracion  →  editar  src\config\settings.json:
::   llm_model         Modelo Ollama             (default: qwen3:8b)
::   llm_base_url      URL base Ollama           (default: http://localhost:11434)
::   unity_port        Puerto TCP Unity          (default: 7777)
::   llm_temperature   Temperatura LLM           (default: 0.2)
::   llm_timeout       Timeout LLM en segundos   (default: 120)
::
:: Ejemplos:
::   start_server.cmd                        <- arranque normal
::   start_server.cmd -c                    <- inicio limpio (archiva logs)
::   start_server.cmd -m qwen3:8b           <- usar modelo concreto
::   start_server.cmd -c -m qwen3:8b      <- inicio limpio + modelo
::
:: Requiere Ollama en marcha y el modelo descargado:  ollama pull qwen3:8b
::
:: ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

:: ── Parsear flags ───────────────────────────────────────────────────────────
set "_CLEAR_FLAG="
set "_MODEL_OVERRIDE="
:parse_flags
if /I "%~1"=="-c"      goto :flag_clear
if /I "%~1"=="--clear" goto :flag_clear
if /I "%~1"=="-m"      goto :flag_model
if /I "%~1"=="--model" goto :flag_model
goto :done_flags
:flag_clear
set "_CLEAR_FLAG=1"
shift
goto :parse_flags
:flag_model
if "%~2"=="" (
    echo [setup] ERROR: -m/--model requiere un nombre de modelo.
    pause
    exit /b 1
)
set "_MODEL_OVERRIDE=%~2"
shift
shift
goto :parse_flags
:done_flags

:: ── Resolver ejecutable Python ──────────────────────────────────────────────
::   1) .venv\ local (dentro del proyecto)
::   2) ..\.venv\ del directorio padre (donde esta actualmente en este entorno)
::   3) Crear .venv local si no existe ninguno
set PY_EXE=.venv\Scripts\python.exe

if not exist "%PY_EXE%" (
    if exist "..\.venv\Scripts\python.exe" (
        set PY_EXE=..\.venv\Scripts\python.exe
        echo [setup] Usando .venv del directorio padre.
    ) else (
        echo.
        echo [setup] No existe .venv. Intentando crearlo...
        where py >nul 2>&1
        if not errorlevel 1 (
            py -3.12 -m venv .venv
        ) else (
            where python >nul 2>&1
            if errorlevel 1 (
                echo [setup] ERROR: No se encontro 'py' ni 'python' en PATH.
                echo         Instala Python 3.11/3.12 y vuelve a ejecutar este lanzador.
                pause
                exit /b 1
            )
            python -m venv .venv
        )
        if errorlevel 1 (
            echo [setup] ERROR: no se pudo crear el entorno virtual .venv
            pause
            exit /b 1
        )
    )
)

if not exist "%PY_EXE%" (
    echo [setup] ERROR: no se encontro %PY_EXE%
    pause
    exit /b 1
)

:: ── Resolver valores desde settings.json ──────────────────────────────────
:: Se usan ficheros temporales para evitar problemas de comillas en CMD.
set LLM_MODEL=qwen3:8b
set UNITY_PORT=7777
set OLLAMA_ROOT=http://localhost:11434

"%PY_EXE%" -c "import json,pathlib; s=json.loads(pathlib.Path(r'src\config\settings.json').read_text(encoding='utf-8')); print(s.get('llm_model','qwen3:8b'))" > "%TEMP%\__spade_m.txt" 2>nul
"%PY_EXE%" -c "import json,pathlib; s=json.loads(pathlib.Path(r'src\config\settings.json').read_text(encoding='utf-8')); print(s.get('unity_port',7777))" > "%TEMP%\__spade_p.txt" 2>nul
"%PY_EXE%" -c "import json,re,pathlib; s=json.loads(pathlib.Path(r'src\config\settings.json').read_text(encoding='utf-8')); u=s.get('llm_base_url','http://localhost:11434'); print(re.sub(r'/v1/?$','',u).rstrip('/'))" > "%TEMP%\__spade_u.txt" 2>nul
if exist "%TEMP%\__spade_m.txt" set /p LLM_MODEL= < "%TEMP%\__spade_m.txt"
if exist "%TEMP%\__spade_p.txt" set /p UNITY_PORT= < "%TEMP%\__spade_p.txt"
if exist "%TEMP%\__spade_u.txt" set /p OLLAMA_ROOT= < "%TEMP%\__spade_u.txt"
if exist "%TEMP%\__spade_m.txt" del "%TEMP%\__spade_m.txt"
if exist "%TEMP%\__spade_p.txt" del "%TEMP%\__spade_p.txt"
if exist "%TEMP%\__spade_u.txt" del "%TEMP%\__spade_u.txt"

if "%LLM_MODEL%"=="" set LLM_MODEL=qwen3:8b
if "%UNITY_PORT%"=="" set UNITY_PORT=7777
if "%OLLAMA_ROOT%"=="" set OLLAMA_ROOT=http://localhost:11434

:: Si se paso -m/--model en linea de comandos, tiene prioridad sobre settings.json
if defined _MODEL_OVERRIDE set "LLM_MODEL=!_MODEL_OVERRIDE!"

:: Exponer el modelo efectivo al proceso Python (override temporal por ejecucion)
set "NPC_LLM_MODEL_OVERRIDE=!LLM_MODEL!"

:: ── Verificar dependencias Python ──────────────────────────────────────────
echo.
echo [setup] Verificando dependencias Python...
"%PY_EXE%" -c "import spade, spade_bdi, spade_llm, agentspeak, pyjabber, networkx, aiohttp" >nul 2>&1
if errorlevel 1 (
    echo [setup] Instalando dependencias de requirements.txt...
    "%PY_EXE%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo [setup] ERROR: fallo instalando dependencias.
        pause
        exit /b 1
    )
) else (
    echo [setup] Dependencias OK.
)

:: ── Verificar Ollama ────────────────────────────────────────────────────────
echo.
echo [setup] Verificando Ollama en !OLLAMA_ROOT!...
curl -s --max-time 5 "!OLLAMA_ROOT!/api/tags" >nul 2>&1
if errorlevel 1 (
    echo [setup] ERROR: Ollama no responde en !OLLAMA_ROOT!
    echo         Abre Ollama Desktop y espera a que diga "Running".
    echo         Para usar otra URL edita llm_base_url en src\config\settings.json
    pause
    exit /b 1
)
echo [setup] Ollama OK.

:: ── Warm-up del modelo ──────────────────────────────────────────────────────
echo.
echo [setup] Modelo  = !LLM_MODEL!
echo [setup] Puerto  = !UNITY_PORT!
echo.
echo [setup] Cargando modelo "!LLM_MODEL!" en RAM ^(warm-up^)...
echo         Esto puede tardar 10-30 s la primera vez...

curl -s --max-time 120 -X POST "!OLLAMA_ROOT!/api/generate" ^
     -H "Content-Type: application/json" ^
     -d "{\"model\":\"!LLM_MODEL!\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":\"30m\"}" >nul 2>&1

if errorlevel 1 (
    echo [setup] AVISO: warm-up no completado. El modelo se cargara en la primera inferencia.
) else (
    echo [setup] Modelo listo en RAM.
)

:: ── Limpiar logs si se paso -c / --clear ───────────────────────────────────
if defined _CLEAR_FLAG (
    echo.
    echo [setup] --clear detectado. Archivando logs de sesiones anteriores...
    for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "_TS=%%T"
    if exist "logs\sessions" (
        if not exist "logs\archive" mkdir "logs\archive"
        for /d %%D in ("logs\sessions\*") do (
            move /y "%%D" "logs\archive\" >nul 2>&1
        )
        for %%F in ("logs\sessions\*.*") do (
            move /y "%%F" "logs\archive\" >nul 2>&1
        )
        echo [setup] Logs movidos a logs\archive\ ^(!_TS!^).
    ) else (
        echo [setup] No existe logs\sessions\, nada que archivar.
    )
    echo.
)

:: ── Arrancar servidor ───────────────────────────────────────────────────────
echo [setup] Iniciando servidor MAS... ^(Ctrl+C para parar^)
echo         Unity debe conectar a 127.0.0.1:!UNITY_PORT!
echo.

"%PY_EXE%" src\main.py
set PY_EXIT=%ERRORLEVEL%

if "%PY_EXIT%"=="0"   exit /b 0
if "%PY_EXIT%"=="130" ( echo [setup] Cierre por Ctrl+C detectado. & exit /b 0 )
if "%PY_EXIT%"=="2"   ( echo [setup] Cierre por Ctrl+C detectado. & exit /b 0 )

echo [setup] El proceso Python finalizo con codigo %PY_EXIT%.
echo         Si se produjo durante Ctrl+C, la terminal puede quedar en estado intermedio;
echo         vuelve a ejecutar en una terminal nueva.
exit /b %PY_EXIT%
