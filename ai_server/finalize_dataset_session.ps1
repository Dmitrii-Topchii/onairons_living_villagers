param(
    [Parameter(Mandatory = $true)]
    [string]$SessionId,
    [int]$MinDurationMs = 700,
    [int]$MaxLatencyMs = 0
)

$SessionDir = Join-Path "data\sessions" $SessionId
$RawPath = Join-Path $SessionDir "interactions_raw.jsonl"
$SftPath = Join-Path $SessionDir "villager_sft_train.jsonl"
$PlotsDir = Join-Path $SessionDir "plots"

if (-not (Test-Path $RawPath)) {
    Write-Error "Could not find raw session log: $RawPath"
    exit 1
}

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

python plot_dataset_stats.py --input $RawPath --output-dir $PlotsDir

$exportArgs = @(
    "export_sft_dataset.py",
    "--input", $RawPath,
    "--output", $SftPath,
    "--session-id", $SessionId,
    "--only-ok",
    "--skip-mock",
    "--require-real-transcript",
    "--min-duration-ms", "$MinDurationMs"
)

if ($MaxLatencyMs -gt 0) {
    $exportArgs += @("--max-latency-ms", "$MaxLatencyMs")
}

python @exportArgs

Write-Host ""
Write-Host "Session finalized:"
Write-Host "  raw:    $RawPath"
Write-Host "  sft:    $SftPath"
Write-Host "  plots:  $PlotsDir"
