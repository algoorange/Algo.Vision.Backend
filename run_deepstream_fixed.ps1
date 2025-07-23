# DeepStream Container Runner Script for Windows (Fixed Version)
# Usage: .\run_deepstream_fixed.ps1

$currentDir = Get-Location

Write-Host "Starting DeepStream container with full project mount..." -ForegroundColor Green
Write-Host "Project Directory: ${currentDir}" -ForegroundColor Yellow

# Create the Docker command mounting the entire project
$dockerCmd = "docker run --gpus all -it --rm " +
    "-v `"${currentDir}:/workspace`" " +
    "-p 8000:8000 " +
    "-p 8554:8554 " +
    "-p 9000:9000 " +
    "--name deepstream-container " +
    "nvcr.io/nvidia/deepstream:7.1-triton-multiarch"

Write-Host "Running command: $dockerCmd" -ForegroundColor Cyan

# Execute the Docker command
Invoke-Expression $dockerCmd

Write-Host "DeepStream container stopped" -ForegroundColor Yellow 