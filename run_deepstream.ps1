# DeepStream Container Runner Script for Windows
# Usage: .\run_deepstream.ps1

$workspaceDir = "deepstream_workspace"
$currentDir = Get-Location

Write-Host "Starting DeepStream container..." -ForegroundColor Green
Write-Host "Workspace: ${currentDir}\${workspaceDir}" -ForegroundColor Yellow

# Run DeepStream container with GPU support and volume mounts
# Mount the entire AlgoVision directory so we can access the app files
docker run --gpus all -it --rm `
    -v "${currentDir}:/workspace" `
    -v "${currentDir}\uploads:/workspace/inputs" `
    -v "${currentDir}\frames:/workspace/outputs" `
    -p 8554:8554 `
    -p 9000:9000 `
    --name deepstream-container `
    nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
    
Write-Host "DeepStream container stopped" -ForegroundColor Yellow
