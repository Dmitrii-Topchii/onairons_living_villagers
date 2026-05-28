param(
    [string]$ModelId = "qwen2-1.5b-instruct",
    [string]$LmStudioBaseUrl = "http://127.0.0.1:1234/v1",
    [string]$SessionId = "",
    [string]$SessionNotes = "local qwen2 stt gameplay dataset session",
    [int]$MaxNewTokens = 50,
    [double]$Temperature = 0.85,
    [double]$TopP = 0.9,
    [string]$SttBackend = "faster_whisper",
    [string]$SttModelId = "tiny.en",
    [string]$SttDevice = "cpu",
    [string]$SttComputeType = "int8",
    [double]$SttTimeoutSeconds = 3.0
)

if (-not $SessionId) {
    $SessionId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
}

$SessionDir = Join-Path "data\sessions" $SessionId
$DatasetPath = Join-Path $SessionDir "interactions_raw.jsonl"
New-Item -ItemType Directory -Force -Path $SessionDir | Out-Null

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$env:LV_SESSION_ID = $SessionId
$env:LV_SESSION_NOTES = $SessionNotes
$env:LV_DATASET_PATH = $DatasetPath

$env:LV_AI_BACKEND = "openai"
$env:LV_OPENAI_BASE_URL = $LmStudioBaseUrl
$env:LV_OPENAI_API_KEY = "not-needed"
$env:LV_MODEL_ID = $ModelId
$env:LV_MAX_NEW_TOKENS = "$MaxNewTokens"
$env:LV_TEMPERATURE = "$Temperature"
$env:LV_TOP_P = "$TopP"

$env:LV_STT_BACKEND = $SttBackend
$env:LV_STT_MODEL_ID = $SttModelId
$env:LV_STT_DEVICE = $SttDevice
$env:LV_STT_COMPUTE_TYPE = $SttComputeType
$env:LV_STT_TIMEOUT_SECONDS = "$SttTimeoutSeconds"
$env:LV_STT_PRELOAD = "true"

Write-Host "Living Villagers dataset session"
Write-Host "  session:  $SessionId"
Write-Host "  model:    $ModelId"
Write-Host "  server:   $LmStudioBaseUrl"
Write-Host "  stt:      $SttBackend / $SttModelId"
Write-Host "  dataset:  $DatasetPath"
Write-Host ""
Write-Host "Keep LM Studio running with the selected model loaded, then launch Minecraft."
Write-Host "Stop this server with Ctrl+C when the data session is done."

python -m uvicorn server:app --host 127.0.0.1 --port 8000
