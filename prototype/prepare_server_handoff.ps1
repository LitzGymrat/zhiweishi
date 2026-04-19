param(
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $scriptDir "dist"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$packageName = "zhiweishi_server_handoff_$timestamp"
$packageDir = Join-Path $OutputRoot $packageName
$zipPath = Join-Path $OutputRoot "$packageName.zip"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (Test-Path $packageDir) {
    Remove-Item -Path $packageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

$filesToCopy = @(
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "app.py",
    "build_knowledge_base.py",
    "compose.public.yml",
    "deploy_public_server.sh",
    "docker-entrypoint.sh",
    "Dockerfile",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "server.env.example",
    "uv.lock"
)

foreach ($relativePath in $filesToCopy) {
    $sourcePath = Join-Path $scriptDir $relativePath
    if (Test-Path $sourcePath) {
        $destinationPath = Join-Path $packageDir $relativePath
        $destinationParent = Split-Path -Parent $destinationPath
        if ($destinationParent) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $destinationPath -Force
    }
}

$pathsToCopy = @(
    "docs",
    "src",
    "data/chroma_db",
    "data/documents",
    "data/real_data"
)

foreach ($relativePath in $pathsToCopy) {
    $sourcePath = Join-Path $scriptDir $relativePath
    if (Test-Path $sourcePath) {
        $destinationPath = Join-Path $packageDir $relativePath
        $destinationParent = Split-Path -Parent $destinationPath
        if ($destinationParent) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
    }
}

$dataFilesToCopy = @(
    "data/README.md",
    "data/bm25_index.json"
)

foreach ($relativePath in $dataFilesToCopy) {
    $sourcePath = Join-Path $scriptDir $relativePath
    if (Test-Path $sourcePath) {
        $destinationPath = Join-Path $packageDir $relativePath
        $destinationParent = Split-Path -Parent $destinationPath
        if ($destinationParent) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $destinationPath -Force
    }
}

$projectDocsRoot = Join-Path (Split-Path $scriptDir -Parent) "docs"
$projectDocsToCopy = @(
    "01_project_brief.md",
    "02_evidence_summary.md",
    "03_technical_design.md",
    "07_competition_positioning.md",
    "10_poc_execution_plan.md",
    "12_poc_onsite_execution.md"
)

foreach ($relativePath in $projectDocsToCopy) {
    $sourcePath = Join-Path $projectDocsRoot $relativePath
    if (Test-Path $sourcePath) {
        $destinationPath = Join-Path $packageDir (Join-Path "project_docs" $relativePath)
        $destinationParent = Split-Path -Parent $destinationPath
        if ($destinationParent) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $destinationPath -Force
    }
}

$pathsToRemove = @(
    "data/case_output",
    "data/documents/uploaded",
    "data/tmp_wheels"
)

foreach ($relativePath in $pathsToRemove) {
    $targetPath = Join-Path $packageDir $relativePath
    if (Test-Path $targetPath) {
        Remove-Item -Path $targetPath -Recurse -Force
    }
}

if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}

Compress-Archive -Path $packageDir -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "部署交付包已生成。" -ForegroundColor Green
Write-Host "目录：$packageDir" -ForegroundColor Green
Write-Host "压缩包：$zipPath" -ForegroundColor Green
Write-Host ""
Write-Host "这个包不包含 .env、.venv、临时上传目录和案例输出目录。" -ForegroundColor Yellow
Write-Host "如果对方负责部署，优先让他看 docs/deploy/部署组员交接说明.md。" -ForegroundColor Yellow
Write-Host "如果对方负责执行 PoC，优先让他看 docs/deploy/PoC执行打包说明.md、project_docs/10_poc_execution_plan.md 和 project_docs/12_poc_onsite_execution.md。" -ForegroundColor Yellow
Write-Host ""