param(
    [int]$Port = 8506
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    $env:app_mode = "poc1_readonly"
    $venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
    $localUrl = "http://localhost:$Port"

    $lanIps = @()
    try {
        $lanIps = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
            Where-Object {
                $_.IPAddress -notlike '127.*' -and
                $_.IPAddress -notlike '169.254.*' -and
                $_.PrefixOrigin -ne 'WellKnown'
            } |
            Select-Object -ExpandProperty IPAddress -Unique
    }
    catch {
        $lanIps = @()
    }

    Write-Host ""
    Write-Host "=== 智维师 PoC1 只读模式 ===" -ForegroundColor Cyan
    Write-Host "本机访问：$localUrl" -ForegroundColor Green
    if ($lanIps.Count -gt 0) {
        Write-Host "局域网访问地址：" -ForegroundColor Yellow
        foreach ($ip in $lanIps) {
            Write-Host "  http://$ip`:$Port"
        }
    }
    else {
        Write-Host "未自动识别到可共享的局域网 IPv4 地址。若现场要共享，建议查看当前网络或直接使用主机热点。" -ForegroundColor Yellow
    }
    Write-Host "已隐藏案例回写和数据导入，适合组员只体验 PoC1。" -ForegroundColor Gray
    Write-Host ""

    Start-Process $localUrl | Out-Null

    if (Test-Path $venvPython) {
        & $venvPython -m streamlit run app.py --server.address 0.0.0.0 --server.port $Port
    }
    else {
        uv run streamlit run app.py --server.address 0.0.0.0 --server.port $Port
    }
}
finally {
    Pop-Location
}