[CmdletBinding()]
param(
    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$Mode = 'pilot',
    [string]$WorkRoot = 'E:\CodexTests\lerobot-rope-4835-20261010',
    [string]$PythonPath = 'E:\CodexTests\lerobot-pr4902-95e6e92\repo\.venv\Scripts\python.exe',
    [int]$Steps = 1000,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$bundleRoot = $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $bundleRoot 'source-manifest.json') -Raw | ConvertFrom-Json
$experimentPath = Join-Path $bundleRoot 'experiment.py'
if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "CUDA Python not found: $PythonPath. Locate the previously verified CUDA environment and pass -PythonPath."
}
if (!(Test-Path -LiteralPath $experimentPath -PathType Leaf)) { throw 'experiment.py is missing.' }
if ($Steps -lt 1) { throw 'Steps must be positive.' }
if (Test-Path -LiteralPath (Join-Path $bundleRoot 'bundle-sha256.json')) {
    $bundleHashes = Get-Content -LiteralPath (Join-Path $bundleRoot 'bundle-sha256.json') -Raw | ConvertFrom-Json
    foreach ($item in $bundleHashes.PSObject.Properties) {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $bundleRoot $item.Name)).Hash.ToLowerInvariant()
        if ($actual -ne $item.Value) { throw "Bundle checksum mismatch: $($item.Name)" }
    }
}

New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
$repoPath = Join-Path $WorkRoot 'source-repo'
$previousSmudge = $env:GIT_LFS_SKIP_SMUDGE
try {
    $env:GIT_LFS_SKIP_SMUDGE = '1'
    if (!(Test-Path -LiteralPath (Join-Path $repoPath '.git'))) {
        if ((Test-Path -LiteralPath $repoPath) -and @(Get-ChildItem -LiteralPath $repoPath -Force).Count -gt 0) {
            throw "Refusing to reuse nonempty directory: $repoPath"
        }
        git init --quiet $repoPath
        if ($LASTEXITCODE -ne 0) { throw 'git init failed.' }
        git -C $repoPath config core.autocrlf false
        if ($LASTEXITCODE -ne 0) { throw 'Local source line-ending configuration failed.' }
        git -C $repoPath remote add origin $manifest.repository
        if ($LASTEXITCODE -ne 0) { throw 'git remote add failed.' }
        git -C $repoPath fetch --depth 1 origin $manifest.pinned_commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned source fetch failed; preserve the directory and report the error.' }
        git -C $repoPath -c advice.detachedHead=false checkout --detach $manifest.pinned_commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned source checkout failed.' }
    }
} finally {
    $env:GIT_LFS_SKIP_SMUDGE = $previousSmudge
}

$actualCommit = (git -C $repoPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $manifest.pinned_commit) { throw 'Unexpected source commit.' }
$repoChanges = @(git -C $repoPath status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $repoChanges.Count -ne 0) { throw 'Source checkout is not clean; inspect without resetting it.' }
$sourcePath = Join-Path $repoPath $manifest.source_path
$sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant()
if ($sourceHash -ne $manifest.source_sha256) { throw 'Pinned world-model source checksum mismatch.' }

$previousCublas = $env:CUBLAS_WORKSPACE_CONFIG
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $preflightLog = Join-Path $WorkRoot 'cuda-preflight.log'
    & $PythonPath (Join-Path $bundleRoot 'cuda_preflight.py') --out (Join-Path $WorkRoot 'cuda-environment.json') 2>&1 | Tee-Object -FilePath $preflightLog
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed.' }
    $resultBase = Join-Path $WorkRoot 'results'
    $resultPath = Join-Path $resultBase $Mode
    $scriptOutput = if ($Mode -eq 'full') { $resultPath } else { $resultBase }
    $experimentArgs = @($experimentPath, '--repo', $repoPath, '--out', $scriptOutput, '--device', 'cuda', '--steps', "$Steps", '--folds', '3', '--seeds', '11,22,33')
    if ($Mode -eq 'smoke') { $experimentArgs += '--smoke' }
    if ($Mode -eq 'pilot') { $experimentArgs += '--pilot' }
    if ($Resume) { $experimentArgs += '--resume' }
    $logPath = Join-Path $WorkRoot "$Mode-console.log"
    & $PythonPath @experimentArgs 2>&1 | Tee-Object -FilePath $logPath
    $trainingExit = $LASTEXITCODE
    if ($trainingExit -ne 0) { throw "Experiment exited with code $trainingExit. Keep partial logs; do not report a complete pass." }
    git -C $repoPath diff --exit-code
    if ($LASTEXITCODE -ne 0) { throw 'Tracked source files changed during the experiment.' }
    Write-Output "Completed $Mode. Results: $resultPath"
} finally {
    $env:CUBLAS_WORKSPACE_CONFIG = $previousCublas
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
}
