$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $PSScriptRoot
$watcher.IncludeSubdirectories = $true
$watcher.Filter = "*.go"
$watcher.EnableRaisingEvents = $true

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] $changeType detected in $path" -ForegroundColor Cyan
    Write-Host "Rebuilding..." -ForegroundColor Yellow
    make
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build Successful!" -ForegroundColor Green
    } else {
        Write-Host "Build Failed!" -ForegroundColor Red
    }
    Write-Host "Watching for changes... (Ctrl+C to stop)"
}

Register-ObjectEvent $watcher "Changed" -Action $action
Register-ObjectEvent $watcher "Created" -Action $action
Register-ObjectEvent $watcher "Deleted" -Action $action
Register-ObjectEvent $watcher "Renamed" -Action $action

Write-Host "Gllama Auto-Builder Started" -ForegroundColor Green
Write-Host "Watching for changes in $PSScriptRoot... (Ctrl+C to stop)"

while ($true) { sleep 1 }
