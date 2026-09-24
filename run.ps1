[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [string]$RuntimeDir = (Join-Path $PSScriptRoot '.runtime'),
    [ValidateRange(1,10)][int]$Limit = 10,
    [string]$Model = 'qwen3.5:2b',
    [string]$Out = '',
    [switch]$Prepare,
    [switch]$Jev
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $RuntimeDir)) { throw 'Run setup.ps1 first.' }
$projectWsl = (& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve project in WSL.' }
$runtimeWsl = (& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $RuntimeDir).Path).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve runtime in WSL.' }
$pipelineArguments = @("$projectWsl/run.py", '--limit', "$Limit", '--model', $Model)
if ($Prepare) { $pipelineArguments += '--prepare' }
if ($Jev) { $pipelineArguments += '--jev' }
if ($Out) {
    if ($Out -match '^[A-Za-z]:[\\/]') {
        $outputPath = (& wsl.exe -d $Distro --exec wslpath -a -u $Out).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve output folder in WSL.' }
    } else { $outputPath = $Out.Replace('\', '/') }
    $pipelineArguments += @('--out', $outputPath)
}
# Forward the optional credential as environment only, never a process argument.
$priorWslenv = $env:WSLENV
try {
    if ($Jev -and $env:AI_GATEWAY_API_KEY) {
        $parts = @($env:WSLENV -split ':' | Where-Object { $_ })
        if (-not ($parts -match '^AI_GATEWAY_API_KEY(?:/.*)?$')) { $parts += 'AI_GATEWAY_API_KEY' }
        $env:WSLENV = $parts -join ':'
    }
    & wsl.exe -d $Distro --cd $projectWsl --exec "$runtimeWsl/venv/bin/python" @pipelineArguments
    $pipelineExit = $LASTEXITCODE
} finally { $env:WSLENV = $priorWslenv }
exit $pipelineExit
