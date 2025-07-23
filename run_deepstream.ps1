# DeepStream Container Runner Script for Windows
# Usage: .\run_deepstream.ps1

$workspaceDir = "deepstream_workspace"
$currentDir = Get-Location

Write-Host "Starting DeepStream container..." -ForegroundColor Green
Write-Host "Workspace: ${currentDir}\${workspaceDir}" -ForegroundColor Yellow

# Create the Docker command as a single string
$dockerCmd = "docker run --gpus all -it --rm " +
    "-v `"${currentDir}\${workspaceDir}:/workspace`" " +
    "-v `"${currentDir}\uploads:/workspace/inputs`" " +
    "-v `"${currentDir}\frames:/workspace/outputs`" " +
    "-p 8554:8554 " +
    "-p 9000:9000 " +
    "--name deepstream-container " +
    "nvcr.io/nvidia/deepstream:7.1-triton-multiarch"

Write-Host "Running command: $dockerCmd" -ForegroundColor Cyan

# Execute the Docker command
Invoke-Expression $dockerCmd

Write-Host "DeepStream container stopped" -ForegroundColor Yellow
