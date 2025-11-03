function Stop-Port {
    param([int]$Port)

    $procIds = (netstat -ano | findstr ":$Port" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Select-Object -Unique) 2>$null

    if ($null -ne $procIds -and $procIds.Count -gt 0) {
        foreach ($procId in $procIds) {
            if ($procId -match '^\d+$') {
                try {
                    taskkill /PID $procId /F | Out-Null
                    Write-Host "✅ Stopped process with PID $procId on port $Port" -ForegroundColor DarkGray
                }
                catch {
                    Write-Host "⚠️  Could not stop PID $procId (may already be closed)" -ForegroundColor DarkYellow
                }
            }
        }
    } else {
        Write-Host "No process found on port $Port" -ForegroundColor Gray
    }
}

