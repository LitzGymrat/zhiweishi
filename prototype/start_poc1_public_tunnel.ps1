param(
    [int]$Port = 8506,
    [int]$ReconnectDelaySeconds = 3
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    Write-Host ""
    Write-Host "=== 智维师临时公网共享模式 ===" -ForegroundColor Cyan
    Write-Host "将先启动 PoC1 只读模式，再用系统自带 SSH 建立临时公网隧道。" -ForegroundColor Gray
    Write-Host "关闭当前窗口后，公网访问地址会失效。" -ForegroundColor Yellow
    Write-Host "临时隧道偶尔会因网络抖动或服务侧回收而断开，脚本会自动重连。" -ForegroundColor Gray
    Write-Host ""

    $localUrl = "http://localhost:$Port"
    $startScript = Join-Path $scriptDir "start_poc1_readonly.ps1"

    Start-Process PowerShell -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", $startScript,
        "-Port", $Port
    ) | Out-Null

    $ready = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $localUrl -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ready) {
        Write-Host "本地服务未能在预期时间内启动，请先单独运行 start_poc1_readonly.bat 检查原型是否可用。" -ForegroundColor Red
        exit 1
    }

    Write-Host "本地只读模式已就绪：$localUrl" -ForegroundColor Green
    Write-Host "正在建立临时公网隧道，请等待输出中的 https 地址。" -ForegroundColor Yellow
    Write-Host "如果后面发生自动重连，请以终端里最新出现的 https 地址为准。" -ForegroundColor Gray
    Write-Host ""

    $sshArgs = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-R", "80:localhost:$Port",
        "nokey@localhost.run",
        "--",
        "--output", "json"
    )

    $lastPublicUrl = $null

    while ($true) {
        Write-Host "正在连接 localhost.run ..." -ForegroundColor Yellow

        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"

            & ssh @sshArgs 2>&1 | ForEach-Object {
                $line = $_.ToString().Trim()
                if ($line) {
                    $event = $null
                    try {
                        $event = $line | ConvertFrom-Json -ErrorAction Stop
                    }
                    catch {
                        $event = $null
                    }

                    if ($null -ne $event) {
                        if ($event.event -eq "authn" -and $event.message) {
                            Write-Host $event.message -ForegroundColor Gray
                        }
                        elseif ($event.event -eq "tcpip-forward" -and $event.address) {
                            $scheme = if ($event.tls_termination) { "https" } else { "http" }
                            $publicUrl = "${scheme}://$($event.address)"

                            if ($publicUrl -ne $lastPublicUrl) {
                                $lastPublicUrl = $publicUrl
                                Write-Host ""
                                Write-Host "当前公网访问地址：$publicUrl" -ForegroundColor Green
                                Write-Host "把这个最新地址发给别人即可。" -ForegroundColor Green
                                Write-Host ""
                            }
                        }
                        elseif ($event.message -and $event.status -ne "success") {
                            Write-Host $event.message -ForegroundColor Yellow
                        }
                    }
                    elseif ($line -match 'permission denied|denied|failed|error|warning') {
                        Write-Host $line -ForegroundColor Yellow
                    }
                }
            }
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        $exitCode = $LASTEXITCODE
        Write-Host ""
        Write-Host "公网隧道已断开，SSH 退出码：$exitCode" -ForegroundColor Yellow
        Write-Host "将在 $ReconnectDelaySeconds 秒后自动重连。按 Ctrl+C 可退出。" -ForegroundColor Gray
        Write-Host ""
        Start-Sleep -Seconds $ReconnectDelaySeconds
    }
}
finally {
    Pop-Location
}