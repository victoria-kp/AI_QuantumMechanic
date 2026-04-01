#!/usr/bin/env bash
# deploy.sh — Deploy AI Quantum Mechanic to a GCP Compute Engine VM.
#
# Prerequisites:
#   1. A GCP account with billing enabled (free $300 credits for new accounts)
#   2. gcloud CLI installed:  brew install --cask google-cloud-sdk
#   3. Authenticated:         gcloud auth login
#   4. Project set:           gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash deploy.sh
#
# Teardown (run when done to avoid charges):
#   bash deploy.sh teardown

set -euo pipefail

# --- Configuration ---
VM_NAME="qm-agent-vm"
ZONE="us-west1-a"
MACHINE_TYPE="e2-medium"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="20GB"
REPO_URL="https://github.com/victoria-kp/AI-QuantumMechanic.git"
PORT="8000"

# --- Teardown ---
if [[ "${1:-}" == "teardown" ]]; then
    echo "Stopping and deleting VM..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    echo "Deleting firewall rule..."
    gcloud compute firewall-rules delete allow-qm-agent --quiet 2>/dev/null || true
    echo "Teardown complete. All billing stopped."
    exit 0
fi

# --- Step 1: Create the VM ---
echo "=== Step 1: Creating VM ($VM_NAME, $MACHINE_TYPE) ==="
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --tags=http-server

# --- Step 2: Open firewall for port 8000 ---
echo "=== Step 2: Opening firewall on port $PORT ==="
gcloud compute firewall-rules create allow-qm-agent \
    --allow=tcp:"$PORT" \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0 \
    --description="Allow access to AI Quantum Mechanic API" \
    2>/dev/null || echo "Firewall rule already exists, skipping."

# --- Step 3: Wait for VM to be ready ---
echo "=== Step 3: Waiting for VM to be ready ==="
sleep 30

# --- Step 4: SSH in and set up Docker + app ---
echo "=== Step 4: Installing Docker and deploying ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    set -euo pipefail

    # Install Docker
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$(. /etc/os-release && echo \$VERSION_CODENAME) stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker \$USER

    # Clone repo and build
    git clone $REPO_URL ai-qm
    cd ai-qm
    sudo docker build -t qm-agent .
    sudo docker run -d \
        --name qm-agent \
        --restart unless-stopped \
        -p $PORT:$PORT \
        qm-agent
"

# --- Step 5: Print the external IP ---
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=========================================="
echo "  Deployed successfully!"
echo "=========================================="
echo ""
echo "  Health check:  curl http://$EXTERNAL_IP:$PORT/health"
echo "  API docs:      http://$EXTERNAL_IP:$PORT/docs"
echo ""
echo "  Test it:"
echo "    curl -X POST http://$EXTERNAL_IP:$PORT/solve \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"api_key\": \"sk-ant-...\", \"problem\": \"Solve the infinite square well.\"}'"
echo ""
echo "  Stop VM (pause billing):   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "  Restart VM:                gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "  Delete everything:         bash deploy.sh teardown"
echo ""
