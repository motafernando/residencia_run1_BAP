param(
    [string]$Manifest = 'configs/smoke.json',
    [string]$ImageTag = 'bias-winobias-v4:0.1',
    [string]$CacheDirectory = '',
    [string]$ResumePath = '',
    [switch]$Inference,
    [switch]$Preflight,
    [switch]$Build
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if ($Inference -and $Preflight) { throw 'Use -Inference ou -Preflight, separadamente.' }
if ($ResumePath -and -not $Inference) { throw 'ResumePath exige -Inference.' }
if ($Build) {
    $lockPath = Join-Path $taskRoot 'image-lock.json'
    if (-not (Test-Path -LiteralPath $lockPath)) { throw 'Execute python lock-image.py antes do build.' }
    $imageLock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
    docker build --build-arg "BASE_IMAGE=$($imageLock.pinned_reference)" -t $imageTag $taskRoot
    if ($LASTEXITCODE -ne 0) { throw 'Falha no build Docker.' }
}
$imageId = docker image inspect --format '{{.Id}}' $imageTag
if ($LASTEXITCODE -ne 0) { throw 'Imagem ausente. Execute este script com -Build.' }
$manifestPath = (Resolve-Path -LiteralPath (Join-Path $taskRoot $Manifest)).Path
$resultsPath = Join-Path $taskRoot 'results'
$cachePath = if ($CacheDirectory) { [IO.Path]::GetFullPath($CacheDirectory) } else { Join-Path $taskRoot 'cache' }
New-Item -ItemType Directory -Force -Path $resultsPath,$cachePath | Out-Null
$dockerArgs = @('run','--rm','--shm-size=1g',
    '--mount',"type=bind,source=$taskRoot\data,target=/app/data,readonly",
    '--mount',"type=bind,source=$manifestPath,target=/app/configs/smoke.json,readonly",
    '--mount',"type=bind,source=$resultsPath,target=/app/results",
    '--mount',"type=bind,source=$cachePath,target=/cache",
    '-e',"EXPERIMENT_IMAGE_ID=$imageId")
if ($env:WINOBIAS_QUEUE_ID) { $dockerArgs += @('--label',"winobias.queue=$($env:WINOBIAS_QUEUE_ID)") }
if ($Inference -or $Preflight) { $dockerArgs += @('--gpus','all') }
if ($env:HF_TOKEN) { $dockerArgs += @('-e','HF_TOKEN') }
if ($Preflight) {
    $preflightPath = Join-Path $taskRoot 'preflight.py'
    $dockerArgs += @('--entrypoint','python','--mount',"type=bind,source=$preflightPath,target=/app/preflight.py,readonly",
                     $imageTag,'/app/preflight.py')
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw 'Falha na checagem de CUDA/tokenizer.' }
    return
}
$dockerArgs += @($imageTag,'--manifest','/app/configs/smoke.json')
if ($ResumePath) {
    $resumeFull = (Resolve-Path -LiteralPath $ResumePath).Path
    $resultsPrefix = [IO.Path]::GetFullPath($resultsPath).TrimEnd('\') + '\'
    if (-not $resumeFull.StartsWith($resultsPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'ResumePath deve estar dentro de results.' }
    $dockerArgs += @('--resume-from', ('/app/results/' + $resumeFull.Substring($resultsPrefix.Length).Replace('\','/')))
}
if (-not $Inference) { $dockerArgs += '--prepare-only' }
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw 'Execução interrompida; consulte o registro em results.' }
