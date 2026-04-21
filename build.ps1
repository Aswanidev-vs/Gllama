$BinDir = "bin"

if (-not (Test-Path -Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

Write-Host "Building Gllama Server..." -ForegroundColor Cyan
go build -o "$BinDir/gllama-server.exe" ./cmd/gllama-server
go build -o "$BinDir/gllama-server" ./cmd/gllama-server

Write-Host "Building Gllama CLI..." -ForegroundColor Cyan
go build -o "$BinDir/gllama.exe" ./cmd/gllama
go build -o "$BinDir/gllama" ./cmd/gllama

Write-Host "Done! Binaries are in the '$BinDir' folder." -ForegroundColor Green
