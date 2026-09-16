@echo off
:: Lanzador de doble clic para run_demo.ps1. Admite los mismos parametros:
::   run_demo.cmd -List
::   run_demo.cmd -Demo CO6
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1" %*
if errorlevel 1 pause
