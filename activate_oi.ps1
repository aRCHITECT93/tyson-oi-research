# OI Research Environment Activator
# Run: . .\activate_oi.ps1

Write-Host ""
Write-Host "  OI Research Environment" -ForegroundColor Cyan
Write-Host "  RTX 5070 Ti | CUDA 12.8 | snntorch 0.9 | brian2 2.6" -ForegroundColor DarkCyan
Write-Host ""

# Activate venv
. "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Force UTF-8 output (fixes box-drawing characters on Windows)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Add project to Python path
$env:PYTHONPATH = $PSScriptRoot

Write-Host "  Commands:" -ForegroundColor Yellow
Write-Host "    python run.py neurons     -- Neuron model comparison"
Write-Host "    python run.py stdp        -- STDP learning rules"
Write-Host "    python run.py reservoir   -- Reservoir computing"
Write-Host "    python run.py pong        -- DishBrain Pong (ESN)"
Write-Host "    python run.py pong --lsm  -- DishBrain Pong (spiking)"
Write-Host "    python run.py scaling     -- Scaling laws (5-15 min)"
Write-Host "    python run.py all         -- Full suite"
Write-Host ""
Write-Host "  Theory docs in: theory\" -ForegroundColor DarkGray
Write-Host "  Results saved to: experiments\results\" -ForegroundColor DarkGray
Write-Host ""
