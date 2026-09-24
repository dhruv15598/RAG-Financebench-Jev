[CmdletBinding()]
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$RuntimeDir = (Join-Path $PSScriptRoot '.runtime'),
    [string]$ModelDir = (Join-Path $PSScriptRoot '.runtime\models\reranker'),
    [string]$Requirements = '',
    [string]$RerankerScript = '',
    [string]$ReportScript = '',
    [string]$Distro = 'Ubuntu',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$shareRoot = (Resolve-Path $PSScriptRoot).Path
$project = (Resolve-Path (Join-Path $shareRoot $ProjectRoot) -ErrorAction SilentlyContinue)
if (-not $project) { $project = (Resolve-Path $ProjectRoot -ErrorAction SilentlyContinue) }
if (-not $project) { throw "ProjectRoot does not exist: $ProjectRoot" }
$project = $project.Path
if (-not $Requirements) { $Requirements = Join-Path $project 'requirements.txt' }
if (-not $RerankerScript) { $RerankerScript = Join-Path $project 'reranker.py' }
if (-not $ReportScript) { $ReportScript = Join-Path $project 'run.py' }

$cfg = [ordered]@{
    ShareRoot = $shareRoot; ProjectRoot = $project; RuntimeDir = $RuntimeDir
    ModelDir = $ModelDir; Requirements = $Requirements; RerankerScript = $RerankerScript
    ReportScript = $ReportScript; OllamaUrl = 'http://127.0.0.1:11435'
    RerankerUrl = 'http://127.0.0.1:11436'
}
if ($CheckOnly) {
    $cfg.RequirementsPresent = Test-Path $Requirements
    $cfg.RerankerPresent = Test-Path $RerankerScript
    $cfg.ReportPresent = Test-Path $ReportScript
    $cfg | ConvertTo-Json -Depth 3
    if (-not ($cfg.RequirementsPresent -and $cfg.RerankerPresent -and $cfg.ReportPresent)) { exit 2 }
    exit 0
}

New-Item -ItemType Directory -Force -Path $RuntimeDir, $ModelDir, (Join-Path $RuntimeDir 'logs') | Out-Null
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw 'WSL is not installed. Install WSL/Ubuntu, reboot if Windows requests it, then rerun.' }
if (-not (Test-Path $Requirements)) { throw "requirements.txt not found: $Requirements" }
$distros = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { ($_ -replace "`0", '').Trim() })
if (-not ($distros -contains $Distro)) {
    Write-Host 'Ubuntu is not installed; requesting the standard WSL Ubuntu install.'
    & wsl.exe --install -d $Distro
    if ($LASTEXITCODE -ne 0) { throw 'WSL Ubuntu installation failed. Install Ubuntu from an elevated PowerShell and rerun.' }
    throw 'WSL Ubuntu installation started. Reboot if Windows requests it, then rerun setup.ps1.'
}

function Invoke-Wsl([string]$Command) {
    & wsl.exe -d $Distro -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "WSL command failed ($LASTEXITCODE): $Command" }
}
function Wait-WslHttp([string]$Uri, [int]$Seconds = 90) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        try { Invoke-Wsl "curl -fsS '$Uri' >/dev/null"; return }
        catch { Start-Sleep -Seconds 2 }
    } while ((Get-Date) -lt $deadline)
    throw "WSL service did not become ready: $Uri"
}
function Convert-ToWsl([string]$Path) {
    $p = (Resolve-Path $Path).Path
    return (& wsl.exe -d $Distro -- wslpath -a -u $p).Trim()
}

$reqWsl = Convert-ToWsl $Requirements
$projectWsl = Convert-ToWsl $project
$runtimeWsl = Convert-ToWsl $RuntimeDir
$modelWsl = Convert-ToWsl $ModelDir
$venvWsl = "$runtimeWsl/venv"

# Bootstrap the small WSL runtime and OCR dependency; never replace an existing
# Ubuntu install. All services below run in this same distro so localhost works.
Invoke-Wsl "sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip git curl tesseract-ocr"
Invoke-Wsl "command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh"
Invoke-Wsl "python3.12 -m venv '$venvWsl' 2>/dev/null || python3 -m venv '$venvWsl'"
Invoke-Wsl "'$venvWsl/bin/python' -m pip install -r '$reqWsl'"

Invoke-Wsl "if ! curl -fsS http://127.0.0.1:11435/api/tags >/dev/null; then nohup env OLLAMA_HOST=127.0.0.1:11435 ollama serve > '$runtimeWsl/logs/ollama.log' 2>&1 & fi"
Wait-WslHttp 'http://127.0.0.1:11435/api/tags'
Invoke-Wsl "OLLAMA_HOST=127.0.0.1:11435 ollama pull 'embeddinggemma:latest'"
Invoke-Wsl "OLLAMA_HOST=127.0.0.1:11435 ollama pull 'qwen3.5:2b'"

$hf = ("'" + $venvWsl + "/bin/python' -c 'from huggingface_hub import snapshot_download; " +
    'snapshot_download(repo_id="Qwen/Qwen3-Reranker-0.6B", revision="e61197ed45024b0ed8a2d74b80b4d909f1255473", local_dir="' + $modelWsl + '")' + "'")
Invoke-Wsl $hf

$rerankerScriptWsl = Convert-ToWsl $RerankerScript
$reportScriptWsl = Convert-ToWsl $ReportScript
Invoke-Wsl "if ! curl -fsS http://127.0.0.1:11436/health >/dev/null; then nohup '$venvWsl/bin/python' '$rerankerScriptWsl' --host 127.0.0.1 --port 11436 --model '$modelWsl' > '$runtimeWsl/logs/reranker.log' 2>&1 & fi"
Wait-WslHttp 'http://127.0.0.1:11436/health'
Invoke-Wsl "'$venvWsl/bin/python' '$reportScriptWsl' --prepare"
Write-Host "Runtime ready. Run the generated notebook/report from $project or rerun with -CheckOnly to inspect resolved paths."
