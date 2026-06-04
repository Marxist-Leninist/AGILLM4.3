param(
  [string]$CoordinatorUrl = $env:AGILLM41_COORDINATOR_URL,
  [string]$JoinCode = $env:AGILLM41_JOIN_CODE,
  [string]$Device = "cpu",
  [int]$Threads = [Math]::Max(1, [Environment]::ProcessorCount / 2)
)

if (-not $CoordinatorUrl) {
  throw "Set AGILLM41_COORDINATOR_URL or pass -CoordinatorUrl"
}

python public_join/agillm41_join_worker.py `
  --coordinator-url $CoordinatorUrl `
  --join-code $JoinCode `
  --device $Device `
  --threads $Threads `
  --loop
