# Update Scientific Data Simulator (Windows PowerShell 5.1+ or PowerShell 7): pull the latest code
# (a git checkout on a branch with an upstream), reinstall the profile chosen at install time,
# and run the doctor.
#
#   .\scripts\update.ps1 [-Setup core|standard|all|dev|ui]   # overrides the remembered profile
param(
    [string]$Setup = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

$venvPython = if ($IsLinux -or $IsMacOS) { ".venv/bin/python" } else { ".venv\Scripts\python.exe" }
if (-not (Test-Path $venvPython)) {
    Write-Host "No .venv found: run scripts\install.cmd first." -ForegroundColor Red
    exit 1
}
if (-not $Setup) {
    $Setup = if (Test-Path ".venv\sds-profile") { (Get-Content ".venv\sds-profile" -TotalCount 1).Trim() } else { "standard" }
}
if (@("core", "standard", "all", "dev", "ui") -notcontains $Setup) {
    Write-Host "Unknown profile '$Setup' (core, standard, all, dev, ui)" -ForegroundColor Red
    exit 2
}

if ((Test-Path ".git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
    $dirty = git status --porcelain --untracked-files=no
    $upstream = $null
    try { $upstream = git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null } catch { $upstream = $null }
    $hasUpstream = ($LASTEXITCODE -eq 0) -and $upstream
    if ($dirty) {
        Write-Host "You have uncommitted changes; not pulling. Commit or stash them, then run again." -ForegroundColor Yellow
    } elseif ($hasUpstream) {
        Write-Host "Pulling the latest code ..."
        git pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Host "git pull failed; resolve it and run again." -ForegroundColor Red; exit 1 }
    } else {
        Write-Host "This branch has no upstream; skipping git pull."
    }
} else {
    Write-Host "Not a git checkout; updating the installed packages only."
}

$extras = if ($Setup -eq "core") { "" } elseif ($Setup -eq "ui") { "[standard]" } else { "[$Setup]" }
Write-Host "Reinstalling profile '$Setup' ..."
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
exit 0
