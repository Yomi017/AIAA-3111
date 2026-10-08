param([string]$PythonExe = 'C:\ProgramData\anaconda3\python.exe', [switch]$Fresh)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if ($Fresh) {
    $taskRoot = Join-Path $PSScriptRoot ('reproduced_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    New-Item -ItemType Directory -Path $taskRoot | Out-Null
    Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*v2.py' | Copy-Item -Destination $taskRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'data') -Destination $taskRoot -Recurse
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'research_v2') -Destination $taskRoot -Recurse
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README_v2.md') -Destination $taskRoot
    New-Item -ItemType Directory -Path (Join-Path $taskRoot 'results_v2') | Out-Null
    foreach ($name in @('frozen_selection.json','protocol_amendment.json','numerical_fix.json')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "results_v2\$name") -Destination (Join-Path $taskRoot 'results_v2')
    }
}
Push-Location -LiteralPath $taskRoot
try {
    $jobs = @(
        @('experiment_v2.py','--run','dev42','--split','development'),
        @('experiment_v2.py','--run','confirm42','--split','confirmation'),
        @('iterate_v2.py','--run','extra42','--split','development','--models','multi'),
        @('iterate_v2.py','--run','extra_confirm','--split','confirmation','--models','multi'),
        @('reconstruction_v2.py','--run','recon42','--split','development'),
        @('reconstruction_v2.py','--run','recon_confirm','--split','confirmation'),
        @('density_v2.py','--run','stable_dev','--split','development'),
        @('density_v2.py','--run','stable_confirm','--split','confirmation'),
        @('experiment_v2.py','--run','confirm7','--split','confirmation','--mode','neural','--seed','7'),
        @('experiment_v2.py','--run','confirm2026','--split','confirmation','--mode','neural','--seed','2026'),
        @('evaluate_frozen_v2.py','--split','development','--runs','dev42','extra42','recon42','stable_dev'),
        @('evaluate_frozen_v2.py'),
        @('detector_v2.py','fit','--method','subspace_lof50'),
        @('verify_exports_v2.py'),
        @('test_protocol_v2.py'),
        @('analyze_results_v2.py'),
        @('make_report_v2.py')
    )
    foreach ($job in $jobs) {
        Write-Host ('Running ' + ($job -join ' '))
        & $PythonExe @job
        if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + ($job -join ' ')) }
    }
    Write-Host ('Completed: ' + $taskRoot)
} finally { Pop-Location }
