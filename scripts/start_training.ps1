param(
    [string]$Config = 'configs/cpu_light.yaml',
    [string]$RunName = 'cpu_light_bce_256',
    [int]$MaxEpochs = 60
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
    throw 'RunName must contain only letters, digits, dots, underscores or hyphens.'
}
if ($MaxEpochs -le 0) { throw 'MaxEpochs must be positive.' }
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $repoRoot '.venv/Scripts/python.exe'
$configPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Config))
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv first.' }
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Config file is missing.' }
$logDir = Join-Path $repoRoot ".tools/training/$RunName"
if (Test-Path -LiteralPath $logDir) { throw 'Run log directory already exists; choose a new RunName.' }
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:NO_ALBUMENTATIONS_UPDATE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TORCH_HOME = Join-Path $repoRoot '.tools/torch'
$env:HF_HOME = Join-Path $repoRoot '.tools/huggingface'
if (Test-Path 'C:\Program Files\Git\cmd') { $env:Path += ';C:\Program Files\Git\cmd' }
$stdout = Join-Path $logDir 'stdout.log'
$stderr = Join-Path $logDir 'stderr.log'
$arguments = @('-u', '-m', 'src.train', '--config', ('"' + $configPath + '"'),
               '--run-name', $RunName, '--max-epochs', $MaxEpochs, '--keep-awake')
$process = Start-Process -FilePath $pythonPath -ArgumentList $arguments `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$metadata = @{
    pid = $process.Id
    started_utc = [DateTime]::UtcNow.ToString('o')
    run_name = $RunName
    config = $configPath
    max_epochs = $MaxEpochs
    stdout = $stdout
    stderr = $stderr
}
$metadata | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $logDir 'process.json')
Start-Sleep -Seconds 3
$process.Refresh()
if ($process.HasExited) {
    Get-Content -LiteralPath $stderr
    throw "Training exited immediately (code $($process.ExitCode))."
}
Write-Output "Started training PID $($process.Id), run $RunName"
Write-Output "Log: $stdout"
Write-Output "Progress: results/runs/$RunName/summary.json"
