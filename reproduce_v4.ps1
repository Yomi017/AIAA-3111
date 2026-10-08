param([string]$PythonExe='python', [string]$ArtifactPythonExe='python', [string]$RunDirectory='')
$ErrorActionPreference='Stop'
if (-not $RunDirectory) { $RunDirectory=Join-Path $PSScriptRoot ('results_v4\replication_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $RunDirectory) { throw 'Choose a NEW replication directory. Previous results must be preserved.' }
Push-Location -LiteralPath $PSScriptRoot
try {
  foreach ($job in @(@('prepare_data_v4.py','--download'),@('audit_v4.py'),@('test_protocol_v2.py'),@('experiment_v4.py','development','--out',$RunDirectory),@('experiment_v4.py','freeze','--out',$RunDirectory),@('experiment_v4.py','confirmation','--out',$RunDirectory),@('test_protocol_v4.py','--run',$RunDirectory),@('export_predictions_v4.py','--run',$RunDirectory),@('analyze_v4.py','--run',$RunDirectory,'--out',(Join-Path $RunDirectory 'analysis')))) {
    & $PythonExe @job
    if ($LASTEXITCODE -ne 0) { throw ('Failed: '+($job -join ' ')) }
  }
  & $ArtifactPythonExe build_report_v4.py --analysis (Join-Path $RunDirectory 'analysis') --run $RunDirectory --output (Join-Path $RunDirectory 'deliverables')
  if ($LASTEXITCODE -ne 0) { throw 'Report builder failed' }
  Write-Host ('Replication complete: '+$RunDirectory)
} finally { Pop-Location }
