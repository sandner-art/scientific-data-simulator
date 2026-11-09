# Install Scientific Data Simulator into .\.venv (Windows PowerShell 5.1+ or PowerShell 7).
#
#   .\scripts\install.ps1 [-Setup core|standard|all|dev|ui]
#
# If scripts are blocked, run scripts\install.cmd instead (it starts this script with
# -ExecutionPolicy Bypass for this run only). Set $env:PYTHON to choose the interpreter.
param(
    [ValidateSet("core", "standard", "all", "dev", "ui")]
    [string]$Setup = "standard"
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Test-Python([string[]]$Command) {
    try {
        $exe = $Command[0]
        $rest = @()
        if ($Command.Length -gt 1) { $rest = $Command[1..($Command.Length - 1)] }
        & $exe @rest -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

$candidates = @()
if ($env:PYTHON) { $candidates += , @($env:PYTHON) }
$candidates += , @("python")      # the one on PATH (the Microsoft Store stub fails the test and is skipped)
$candidates += , @("py", "-3")    # the Python launcher: newest registered Python
$candidates += , @("python3")
$python = $null
foreach ($c in $candidates) {
    if (Test-Python $c) { $python = $c; break }
}
if (-not $python) {
    Write-Host "Python 3.11 or newer was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this again."
    exit 1
}
$pyExe = $python[0]
$pyArgs = @()
if ($python.Length -gt 1) { $pyArgs = $python[1..($python.Length - 1)] }
& $pyExe @pyArgs -c "import sys; print('Using', sys.executable, sys.version.split()[0])"

$venvPython = if ($IsLinux -or $IsMacOS) { ".venv/bin/python" } else { ".venv\Scripts\python.exe" }
function Test-Venv {  # a working environment on Python 3.11 or newer
    if (-not (Test-Path $venvPython)) { return $false }
    & $venvPython -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $venvPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}
if ((Test-Path ".venv") -and -not (Test-Venv)) {
    Write-Host "Replacing .venv (incomplete, or made with a Python older than 3.11) ..."
    Remove-Item -Recurse -Force ".venv"
}
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv ..."
    & $pyExe @pyArgs -m venv .venv
    if (($LASTEXITCODE -ne 0) -or -not (Test-Venv)) {
        if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
        Write-Host "Could not create the virtual environment." -ForegroundColor Red
        exit 1
    }
}

$extras = if ($Setup -eq "core") { "" } elseif ($Setup -eq "ui") { "[standard]" } else { "[$Setup]" }
Write-Host "Installing profile '$Setup' ..."
& $venvPython -m pip install --upgrade pip | Out-Null
& $venvPython -m pip install -e ".$extras"
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed." -ForegroundColor Red; exit 1 }
if ($Setup -eq "ui") {
    & $venvPython -m pip install -e ./ui
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install of the UI failed." -ForegroundColor Red; exit 1 }
}
Set-Content -Path ".venv\sds-profile" -Value $Setup

Write-Host ""
& $venvPython -m simulator doctor
Write-Host ""
if ($IsLinux -or $IsMacOS) {
    Write-Host "Done. Activate with:  . .venv/bin/Activate.ps1"
} else {
    Write-Host "Done. Activate with:  .venv\Scripts\Activate.ps1   (cmd: .venv\Scripts\activate.bat)"
}
Write-Host "Then try:             sds examples list"
if ($Setup -eq "ui") { Write-Host "Start the UI:         sds ui      (or double-click scripts\start-ui.cmd)" }
exit 0
