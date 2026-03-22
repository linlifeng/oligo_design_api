# AWS micro instance tweaking

## 1. Memory Fix: Add swap space
The t2.micro only has 1GB of RAM; it may freeze when VS Code Remote-SSH connects.

```bash
# Create 2GB swap file
sudo dd if=/dev/zero of=/swapfile bs=128M count=16
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
sudo swapon --show
free -h
```

## 2. Disk Space Issue: Resize EBS volume
The root volume may be 8GB (instead of 30GB). Steps:

### AWS Console
1. Stop EC2 instance
2. Go to Volumes → select root volume
3. Actions → Modify Volume → set size to 30GB
4. Start instance

### After instance starts
```bash
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1

df -h /
```

## 3. DNS automation (Cloudflare auto-update)
Use this if EC2 public IPv4 changes at each stop/start.

### 3.1 Create Cloudflare API token
- Manage tokens → Zone → DNS → Edit
- Resource: oligodesign.com

### 3.2 Script: `/home/ubuntu/update-dns.sh`

```bash
#!/bin/bash

# Variables (replace with your values)
ZONE_ID="your-zone-id-here"
DNS_NAME="api.oligodesign.com"
API_TOKEN="your-api-token-here"

# Get AWS metadata token
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Get current public IP
PUBLIC_IP=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4)

# Get DNS record ID
RECORD_ID=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?type=A&name=$DNS_NAME" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" | jq -r '.result[0].id')

# Update DNS record
curl -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$RECORD_ID" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  --data "{\"type\": \"A\", \"name\": \"$DNS_NAME\", \"content\": \"$PUBLIC_IP\", \"ttl\": 120, \"proxied\": false}"
```

### 3.3 Install dependencies + permissions
```bash
sudo apt install jq -y
chmod +x /home/ubuntu/update-dns.sh
```

### 3.4 Systemd service: `/etc/systemd/system/cloudflare-update.service`

```ini
[Unit]
Description=Update Cloudflare DNS on boot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/home/ubuntu/update-dns.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

### 3.5 Enable and start service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflare-update.service
sudo systemctl start cloudflare-update.service
sudo systemctl status cloudflare-update.service
sudo journalctl -u cloudflare-update.service
```

## 4. SSH configuration

### 4.1 Cloudflare DNS mode
- Orange cloud = proxy active (no SSH)
- Gray cloud = DNS only (SSH works)

### 4.2 Local SSH config (`~/.ssh/config`)

```text
Host api
  HostName api.oligodesign.com
  User ubuntu
  IdentityFile ~/.ssh/your-key.pem
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

### 4.3 Windows DNS cache

```powershell
ipconfig /flushdns
Clear-DnsClientCache
```

## 5. Security groups
| Type | Port | Source | Purpose |
|------|------|--------|--------|
| SSH  | 22   | 0.0.0.0/0 (or your IP) | Remote access |
| HTTP | 80   | 0.0.0.0/0 | Web traffic |
| HTTPS| 443  | 0.0.0.0/0 | Web traffic |

## 6. Installed packages
```bash
sudo apt install jq -y
# Docker (optional)
# sudo apt install docker.io -y
# sudo apt install docker-compose -y
# sudo systemctl enable docker
# sudo usermod -aG docker $USER
```

## 7. Verification commands
- `df -h`
- `free -h`
- `sudo systemctl status cloudflare-update.service`
- `sudo journalctl -u cloudflare-update.service --no-pager`
- `curl -s ifconfig.me`
- `dig api.oligodesign.com`
- `ssh api`

## 8. Common issues and fixes
- Swap/VS Code freezes: add swap, upgrade instance
- SSH timeouts: gray cloud DNS + open 22 + flush DNS
- DNS stale: wait 5-15 mins or run service manually
- Disk full: `sudo apt clean && sudo apt autoremove -y`; `sudo journalctl --vacuum-size=100M`

## 9. Future improvements
- Install Docker/Docker Compose
- Set container memory limits (t2.micro)
- Configure log rotation
- Separate subdomains:
  - `ssh.oligodesign.com` (gray cloud for SSH)
  - `api.oligodesign.com` (orange cloud for API)

## 10. Useful aliases (`~/.bashrc`)
```bash
alias dfh='df -h'
alias freeh='free -h'
alias docker-clean='docker system prune -af'
alias ec2-status='sudo systemctl status cloudflare-update.service'
alias ec2-logs='sudo journalctl -u cloudflare-update.service --no-pager | tail -20'
```

> **Before committing**: remove real IPs, API tokens, zone IDs, and any secrets.
> Add this file to `.gitignore` if it includes private config.
