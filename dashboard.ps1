[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [string]$RuntimeDir = (Join-Path $PSScriptRoot '.runtime'),
    [ValidateRange(1024,65535)][int]$Port = 7860
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $RuntimeDir)) { throw 'Run setup.ps1 first.' }
$projectWsl = (& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve project in WSL.' }
$runtimeWsl = (& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $RuntimeDir).Path).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve runtime in WSL.' }
$priorWslenv = $env:WSLENV
try {
    if ($env:AI_GATEWAY_API_KEY) {
        $parts = @($env:WSLENV -split ':' | Where-Object { $_ })
        if (-not ($parts -match '^AI_GATEWAY_API_KEY(?:/.*)?$')) { $parts += 'AI_GATEWAY_API_KEY' }
        $env:WSLENV = $parts -join ':'
    }
    Write-Host "Open http://localhost:$Port once the server is ready. Ctrl+C stops it."
    & wsl.exe -d $Distro --cd $projectWsl --exec "$runtimeWsl/venv/bin/python" "$projectWsl/dashboard.py" --port $Port
} finally { $env:WSLENV = $priorWslenv }
