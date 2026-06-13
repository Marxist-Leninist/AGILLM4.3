param(
  [string]$CoordinatorUrl = $env:AGILLM41_COORDINATOR_URL,
  [string]$JoinCode = $env:AGILLM41_JOIN_CODE,
  [string]$Device = "cpu",
  [int]$Threads = [Math]::Max(1, [int]([Environment]::ProcessorCount / 2)),
  [string]$WorkDir = $env:AGILLM41_JOIN_WORKDIR
)

if (-not $CoordinatorUrl) {
  throw "Set AGILLM41_COORDINATOR_URL or pass -CoordinatorUrl"
}

if (-not $WorkDir) {
  $WorkDir = Join-Path $PWD "agillm41_join_work"
}

$argsList = @(
  "public_join/agillm41_join_worker.py",
  "--coordinator-url", $CoordinatorUrl,
  "--device", $Device,
  "--threads", [string]$Threads,
  "--workdir", $WorkDir,
  "--loop"
)

if ($JoinCode) {
  $argsList += @("--join-code", $JoinCode)
}

python @argsList
