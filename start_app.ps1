# ==================== CineScope Bootstrap (Start) ====================
# Project layout:
#   MovieRecommend/
#     backend/   -> FastAPI (app.py)
#     frontend/  -> index.html (and other static files)
# =====================================================================

# ----- Settings -----
$ProjectRoot = "C:\Users\shaik\Documents\MovieRecommend"
$Backend     = Join-Path $ProjectRoot "backend"
$Frontend    = Join-Path $ProjectRoot "frontend"
$VenvDir     = Join-Path $Backend ".venv"
$PyExe       = Join-Path $VenvDir "Scripts\python.exe"

$ApiHost     = "127.0.0.1"
$ApiPort     = 8000
$WebPort     = 5500

# ----- Helpers -----
function Stop-Port {
    param([int]$Port)

    $procIds = (netstat -ano | findstr ":$Port" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Select-Object -Unique) 2>$null

    if ($procIds) {
        foreach ($procId in @($procIds)) {
            if ($procId -match '^\d+$') {
                try {
                    taskkill /PID $procId /F | Out-Null
                    Write-Host "Stopped PID $procId on port $Port" -ForegroundColor DarkGray
                } catch {
                    Write-Host "Could not stop PID $procId (already closed)" -ForegroundColor DarkYellow
                }
            }
        }
    } else {
        Write-Host "No process found on port $Port" -ForegroundColor Gray
    }
}

function Ensure-Venv {
    if (-not (Test-Path $PyExe)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        if (Get-Command py -ErrorAction SilentlyContinue) {
            py -3 -m venv $VenvDir
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -m venv $VenvDir
        } else {
            throw "Python launcher not found. Install Python 3 and ensure it's on PATH."
        }
    }
}

function Ensure-Requirements {
    $ReqFile = Join-Path $Backend "requirements.txt"
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    & $PyExe -m pip install --upgrade pip
    if (Test-Path $ReqFile) {
        & $PyExe -m pip install -r $ReqFile
    } else {
        & $PyExe -m pip install fastapi uvicorn xgboost numpy pandas scikit-learn
    }
}

# ----- Main Execution -----
Write-Host "== CineScope Bootstrap ==" -ForegroundColor Yellow
Set-Location $ProjectRoot

try {
    # 1️⃣ venv + deps
    Ensure-Venv
    Ensure-Requirements

    # 2️⃣ free ports
    Write-Host "Freeing ports $ApiPort and $WebPort..." -ForegroundColor Cyan
    Stop-Port -Port $ApiPort
    Stop-Port -Port $WebPort

    # 3️⃣ backend
    $BackendUrl = "http://$ApiHost`:$ApiPort"
    Write-Host "Starting backend at $BackendUrl ..." -ForegroundColor Green
    Start-Process -WindowStyle Minimized -WorkingDirectory $Backend `
        -FilePath $PyExe -ArgumentList @("-m","uvicorn","app:app","--reload","--host",$ApiHost,"--port",$ApiPort)

    # 4️⃣ frontend
    $FrontendUrl = "http://127.0.0.1`:$WebPort/frontend/index.html"
    Write-Host "Starting static server at http://127.0.0.1`:$WebPort ..." -ForegroundColor Green
    Start-Process -WindowStyle Minimized -WorkingDirectory $ProjectRoot `
        -FilePath $PyExe -ArgumentList @("-m","http.server","$WebPort")

    Start-Sleep -Seconds 2

    # 5️⃣ open browser
    Start-Process "$BackendUrl/health"
    Start-Process $FrontendUrl

    Write-Host "✅ Backend & Frontend running successfully!" -ForegroundColor Yellow
    Write-Host "🌐 Backend:  $BackendUrl"
    Write-Host "💻 Frontend: $FrontendUrl"
}
catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}



