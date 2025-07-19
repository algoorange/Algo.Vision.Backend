#!/bin/bash

# DeepStream SDK 7.1 Installation Script for WSL2 Ubuntu 22.04
# This script installs DeepStream SDK with all dependencies

set -e

echo "🚀 Starting DeepStream SDK 7.1 Installation for WSL2..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in WSL2
check_wsl2() {
    if [[ ! -f "/proc/version" ]] || ! grep -qi microsoft /proc/version; then
        print_error "This script is designed for WSL2. Please run it in WSL2 environment."
        exit 1
    fi
    print_status "WSL2 environment detected ✓"
}

# Verify NVIDIA GPU access
verify_gpu() {
    print_status "Verifying NVIDIA GPU access..."
    if ! nvidia-smi &>/dev/null; then
        print_error "nvidia-smi not accessible. Please ensure NVIDIA drivers are installed on Windows host."
        print_error "Required: GameReady Driver version 546.65 or newer"
        exit 1
    fi
    print_status "NVIDIA GPU access verified ✓"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
}

# Install prerequisite packages
install_prerequisites() {
    print_status "Installing prerequisite packages..."
    sudo apt-get install -y \
        libssl3 \
        libssl-dev \
        libgles2-mesa-dev \
        libgstreamer1.0-0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer-plugins-base1.0-dev \
        libgstrtspserver-1.0-0 \
        libjansson4 \
        libyaml-cpp-dev \
        libjsoncpp-dev \
        protobuf-compiler \
        gcc \
        make \
        git \
        python3 \
        python3-pip \
        curl \
        wget \
        ca-certificates \
        gnupg \
        lsb-release
}

# Install Docker Engine
install_docker() {
    print_status "Installing Docker Engine..."
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up the repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start docker service
    sudo service docker start
    
    print_status "Testing Docker installation..."
    sudo docker run --rm hello-world
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    print_status "Installing NVIDIA Container Toolkit..."
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
}

# Install X11 utilities for display
install_x11_utils() {
    print_status "Installing X11 utilities..."
    sudo apt-get install -y x11-xserver-utils
}

# Pull DeepStream Docker container
pull_deepstream_container() {
    print_status "Pulling DeepStream 7.1 Docker container..."
    sudo docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch
}

# Create DeepStream workspace directory
create_workspace() {
    print_status "Creating DeepStream workspace..."
    mkdir -p $HOME/deepstream-workspace
    cd $HOME/deepstream-workspace
    
    # Create sample configuration
    cat > run_deepstream.sh << 'EOF'
#!/bin/bash
# DeepStream container runner script

# Enable X11 forwarding
xhost +

# Run DeepStream container
sudo docker run -it --privileged --rm \
    --name=deepstream-container \
    --net=host \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e CUDA_CACHE_DISABLE=0 \
    --device /dev/snd \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v $HOME/deepstream-workspace:/workspace \
    nvcr.io/nvidia/deepstream:7.1-triton-multiarch
EOF
    
    chmod +x run_deepstream.sh
    print_status "DeepStream runner script created at $HOME/deepstream-workspace/run_deepstream.sh"
}

# Test DeepStream installation
test_deepstream() {
    print_status "Testing DeepStream installation..."
    
    # Enable X11 forwarding
    xhost +
    
    # Test DeepStream version
    sudo docker run --rm --gpus all \
        nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
        deepstream-app --version
    
    print_status "DeepStream installation test completed!"
}

# Create Python development environment
setup_python_env() {
    print_status "Setting up Python development environment..."
    
    # Install Python packages for DeepStream development
    pip3 install --user \
        numpy \
        opencv-python \
        pillow \
        pycairo \
        PyGObject \
        requests \
        asyncio \
        websockets
    
    # Create virtual environment for DeepStream development
    python3 -m venv $HOME/deepstream-workspace/venv
    source $HOME/deepstream-workspace/venv/bin/activate
    
    pip install \
        numpy \
        opencv-python \
        pillow \
        pycairo \
        PyGObject \
        requests \
        asyncio \
        websockets \
        fastapi \
        uvicorn
        
    deactivate
    
    print_status "Python environment setup completed!"
}

# Main installation function
main() {
    print_status "Starting DeepStream SDK 7.1 installation..."
    
    check_wsl2
    verify_gpu
    update_system
    install_prerequisites
    install_docker
    install_nvidia_container_toolkit
    install_x11_utils
    pull_deepstream_container
    create_workspace
    setup_python_env
    test_deepstream
    
    print_status "🎉 DeepStream SDK 7.1 installation completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Restart your WSL session: exit and then 'wsl -d Ubuntu-22.04'"
    echo "2. Navigate to workspace: cd ~/deepstream-workspace"
    echo "3. Run DeepStream: ./run_deepstream.sh"
    echo "4. Test with: deepstream-app --version"
    echo
    print_warning "Note: You may need to restart WSL to apply all changes."
}

# Run main function
main "$@" 