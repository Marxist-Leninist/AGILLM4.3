param(
  [string]$GethHost = "5.75.217.57",
  [string]$GethUser = "root",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\agillm35_laptop_reverse_ed25519",
  [string]$RemoteRoot = "/root/agillm41_opportunistic",
  [string]$LocalRoot = "C:\agillm41_worker",
  [string]$WorkerId = "laptop-auto",
  [string]$Python = "python",
  [string]$DirectMLPython = "C:\agillm41_worker\dml_venv\Scripts\python.exe",
  [string]$Device = "auto",
  [int]$Threads = 0,
  [int]$LaneTimeoutSeconds = 300,
  [int]$PollSeconds = 300,
  [switch]$Once,
  [switch]$Force,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sshBase = @("-i", $KeyPath, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3")
$scpBase = @("-i", $KeyPath, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=6")
$remote = "$GethUser@$GethHost"

function Resolve-Threads {
  if ($Threads -gt 0) { return $Threads }
  try {
    $logical = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    if ($logical -and $logical -gt 0) { return [int]$logical }
  } catch {}
  return [Math]::Max(1, [Environment]::ProcessorCount)
}

$resolvedThreads = Resolve-Threads

function New-Dirs {
  foreach ($name in @("code", "runtime", "packages", "updates", "logs", "state")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $LocalRoot $name) | Out-Null
  }
}

function Invoke-SSH([string]$Command) {
  & ssh @sshBase $remote $Command
  if ($LASTEXITCODE -ne 0) { throw "ssh failed rc=$LASTEXITCODE command=$Command" }
}

function Read-SSH([string]$Command) {
  $out = & ssh @sshBase $remote $Command
  if ($LASTEXITCODE -ne 0) { return "" }
  return ($out -join "`n")
}

function Invoke-SCPFrom([string]$RemotePath, [string]$LocalPath) {
  & scp @scpBase "${remote}:$RemotePath" $LocalPath
  if ($LASTEXITCODE -ne 0) { throw "scp from $RemotePath failed rc=$LASTEXITCODE" }
}

function Invoke-SCPTo([string]$LocalPath, [string]$RemotePath) {
  & scp @scpBase $LocalPath "${remote}:$RemotePath"
  if ($LASTEXITCODE -ne 0) { throw "scp to $RemotePath failed rc=$LASTEXITCODE" }
}

function Write-Heartbeat([string]$State, [hashtable]$Extra = @{}) {
  $hb = @{
    worker_id = $WorkerId
    host = $env:COMPUTERNAME
    state = $State
    device = $Device
    threads = $resolvedThreads
    at = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  }
  foreach ($k in $Extra.Keys) { $hb[$k] = $Extra[$k] }
  $tmp = Join-Path $LocalRoot "state\heartbeat.json"
  ($hb | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $tmp -Encoding ascii
  Invoke-SSH "mkdir -p '$RemoteRoot/heartbeats'"
  Invoke-SCPTo $tmp "$RemoteRoot/heartbeats/$WorkerId.json.tmp"
  Invoke-SSH "mv '$RemoteRoot/heartbeats/$WorkerId.json.tmp' '$RemoteRoot/heartbeats/$WorkerId.json'"
}

function Get-RemoteStat([string]$Path) {
  $text = Read-SSH "if [ -s '$Path' ]; then stat -c '%Y %s' '$Path'; fi"
  if (-not $text.Trim()) { return $null }
  $parts = $text.Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
  if ($parts.Count -lt 2) { return $null }
  return @{ Stamp = "$($parts[0])-$($parts[1])"; Mtime = $parts[0]; Size = [int64]$parts[1] }
}

function Copy-IfSizeDiffers([string]$RemotePath, [string]$LocalPath) {
  $stat = Get-RemoteStat $RemotePath
  if ($null -eq $stat) { throw "missing remote path $RemotePath" }
  $stampPath = "$LocalPath.stamp"
  if (
    (Test-Path -LiteralPath $LocalPath) -and
    (Test-Path -LiteralPath $stampPath) -and
    ((Get-Content -LiteralPath $stampPath -Raw).Trim() -eq $stat.Stamp)
  ) {
    return
  }
  Invoke-SCPFrom $RemotePath $LocalPath
  Set-Content -LiteralPath $stampPath -Encoding ascii -Value $stat.Stamp
}

function Get-DeviceLanes {
  if ($Device -eq "auto") {
    $lanes = @(
      @{ Device = "cuda:0"; Suffix = "cuda0"; Python = $Python },
      @{ Device = "directml:1"; Suffix = "igpu"; Python = $DirectMLPython },
      @{ Device = "cpu"; Suffix = "cpu"; Python = $Python }
    )
    return @($lanes | Where-Object { ($_.Device -notlike "directml:*") -or (Test-Path -LiteralPath $_.Python) })
  }
  $suffix = ($Device -replace "[:\\\/\s]", "_")
  $pythonExe = if ($Device -like "directml:*" -or $Device -eq "directml" -or $Device -eq "dml" -or $Device -eq "igpu") { $DirectMLPython } else { $Python }
  return @(@{ Device = $Device; Suffix = $suffix; Python = $pythonExe })
}

function Run-Worker([hashtable]$Lane, [string]$LeaseStamp) {
  $RunDevice = [string]$Lane.Device
  $laneSuffix = [string]$Lane.Suffix
  $pythonExe = [string]$Lane.Python
  if (($RunDevice -like "directml:*" -or $RunDevice -eq "directml" -or $RunDevice -eq "dml" -or $RunDevice -eq "igpu") -and (-not (Test-Path -LiteralPath $pythonExe))) {
    Write-Heartbeat "worker_skipped" @{ lease = $LeaseStamp; run_device = $RunDevice; lane = $laneSuffix; reason = "missing_directml_python"; python = $pythonExe }
    return $null
  }
  $worker = Join-Path $LocalRoot "code\agillm4_slice_bench_worker.py"
  $package = Join-Path $LocalRoot "packages\lease_$WorkerId.pt"
  $shared = Join-Path $LocalRoot "packages\shared_frozen.pt"
  $runtime = Join-Path $LocalRoot "runtime\agillm41.py"
  $out = Join-Path $LocalRoot "updates\$WorkerId`_$LeaseStamp`_$laneSuffix.pt"
  $log = Join-Path $LocalRoot "logs\$WorkerId`_$LeaseStamp`_$laneSuffix.log"
  Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath "$out.tmp" -ErrorAction SilentlyContinue
  $env:OMP_NUM_THREADS = [string]$resolvedThreads
  $env:MKL_NUM_THREADS = [string]$resolvedThreads
  $env:OPENBLAS_NUM_THREADS = [string]$resolvedThreads
  $env:PYTHONWARNINGS = "ignore::FutureWarning"
  $args = @(
    "-u", $worker,
    "--package", $package,
    "--shared", $shared,
    "--runtime", $runtime,
    "--out", $out,
    "--device", $RunDevice,
    "--threads", [string]$resolvedThreads,
    "--worker-id", "$WorkerId-$laneSuffix"
  )
  Write-Heartbeat "running" @{ lease = $LeaseStamp; run_device = $RunDevice; lane = $laneSuffix; python = $pythonExe }
  $oldErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    if ($LaneTimeoutSeconds -gt 0) {
      $job = Start-Job -ScriptBlock {
        param([string]$JobPython, [string[]]$JobArgs, [string]$JobLog)
        & $JobPython @JobArgs > $JobLog 2>&1
        return $LASTEXITCODE
      } -ArgumentList $pythonExe, $args, $log
      $completed = Wait-Job $job -Timeout $LaneTimeoutSeconds
      if ($null -ne $completed) {
        $jobResult = Receive-Job $job
        $rc = if ($jobResult -is [array]) { [int]$jobResult[-1] } else { [int]$jobResult }
      } else {
        Stop-Job $job -ErrorAction SilentlyContinue
        Add-Content -LiteralPath $log -Encoding ascii -Value "lane timed out after $LaneTimeoutSeconds seconds"
        $rc = 124
      }
      Remove-Job $job -Force -ErrorAction SilentlyContinue
    } else {
      & $pythonExe @args > $log 2>&1
      $rc = $LASTEXITCODE
    }
  } finally {
    $ErrorActionPreference = $oldErrorActionPreference
  }
  if ($rc -ne 0) {
    Write-Heartbeat "worker_failed" @{ lease = $LeaseStamp; run_device = $RunDevice; lane = $laneSuffix; rc = $rc; log = $log }
    return $null
  }
  if (-not (Test-Path -LiteralPath $out)) { throw "worker completed but did not create $out" }
  return $out
}

function One-Poll {
  New-Dirs
  if (-not (Test-Path -LiteralPath $KeyPath)) { throw "missing SSH key $KeyPath" }
  Invoke-SSH "mkdir -p '$RemoteRoot/updates' '$RemoteRoot/heartbeats'"
  $leaseRemote = "$RemoteRoot/current/lease_$WorkerId.pt"
  $leaseStat = Get-RemoteStat $leaseRemote
  if ($null -eq $leaseStat) {
    Write-Heartbeat "idle_no_lease"
    return
  }
  $leaseStamp = $leaseStat.Stamp
  $doneFile = Join-Path $LocalRoot "state\done_$WorkerId.txt"
  if ((-not $Force) -and (Test-Path -LiteralPath $doneFile) -and ((Get-Content -LiteralPath $doneFile -Raw).Trim() -eq $leaseStamp)) {
    Write-Heartbeat "idle_already_done" @{ lease = $leaseStamp }
    return
  }

  Write-Heartbeat "pulling" @{ lease = $leaseStamp }
  Copy-IfSizeDiffers "$RemoteRoot/runtime/agillm41.py" (Join-Path $LocalRoot "runtime\agillm41.py")
  foreach ($runtimeName in @("dblocks_train.py", "fused_ce.py", "anchor_memory.py")) {
    $runtimeStat = Get-RemoteStat "$RemoteRoot/runtime/$runtimeName"
    if ($null -ne $runtimeStat) {
      Copy-IfSizeDiffers "$RemoteRoot/runtime/$runtimeName" (Join-Path $LocalRoot "runtime\$runtimeName")
    }
  }
  Copy-IfSizeDiffers "$RemoteRoot/code/agillm4_slice_bench_worker.py" (Join-Path $LocalRoot "code\agillm4_slice_bench_worker.py")
  Copy-IfSizeDiffers "$RemoteRoot/current/shared_frozen.pt" (Join-Path $LocalRoot "packages\shared_frozen.pt")
  Copy-IfSizeDiffers $leaseRemote (Join-Path $LocalRoot "packages\lease_$WorkerId.pt")

  if ($DryRun) {
    Write-Heartbeat "dry_run_ready" @{ lease = $leaseStamp }
    return
  }

  $uploaded = @()
  foreach ($lane in (Get-DeviceLanes)) {
    $out = Run-Worker $lane $leaseStamp
    if ($null -eq $out) { continue }
    $laneSuffix = [string]$lane.Suffix
    Write-Heartbeat "uploading" @{ lease = $leaseStamp; lane = $laneSuffix; bytes = (Get-Item -LiteralPath $out).Length }
    $remoteOut = "$RemoteRoot/updates/$WorkerId`_$leaseStamp`_$laneSuffix.pt"
    Invoke-SCPTo $out "$remoteOut.tmp"
    Invoke-SSH "mv '$remoteOut.tmp' '$remoteOut'"
    $uploaded += $remoteOut
  }
  if ($uploaded.Count -eq 0) { throw "all requested devices failed for lease $leaseStamp" }
  Set-Content -LiteralPath $doneFile -Encoding ascii -Value $leaseStamp
  Write-Heartbeat "done" @{ lease = $leaseStamp; remote_updates = $uploaded; count = $uploaded.Count }
}

$lockPath = Join-Path $LocalRoot "state\worker.lock"
New-Dirs
try {
  New-Item -ItemType Directory -Path $lockPath -ErrorAction Stop | Out-Null
} catch {
  try {
    $existing = Get-Content -LiteralPath (Join-Path $lockPath "owner.json") -Raw | ConvertFrom-Json
    $pid = [int]$existing.pid
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
      Write-Output "AGILLM4.1 opportunistic worker already running as PID $pid"
      exit 0
    }
  } catch {
    Write-Output "AGILLM4.1 opportunistic worker lock exists at $lockPath"
    exit 0
  }
}
@{
  pid = $PID
  worker_id = $WorkerId
  started_at = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lockPath "owner.json") -Encoding ascii

try {
while ($true) {
  try {
    One-Poll
  } catch {
    try { Write-Heartbeat "error" @{ error = "$_" } } catch {}
    Write-Error $_
    if ($Once) { exit 1 }
  }
  if ($Once) { break }
  Start-Sleep -Seconds $PollSeconds
}
} finally {
  try {
    $ownerPath = Join-Path $lockPath "owner.json"
    if (Test-Path -LiteralPath $ownerPath) {
      $owner = Get-Content -LiteralPath $ownerPath -Raw | ConvertFrom-Json
      if ([int]$owner.pid -eq $PID) {
        Remove-Item -LiteralPath $lockPath -Recurse -Force -ErrorAction SilentlyContinue
      }
    }
  } catch {}
}
