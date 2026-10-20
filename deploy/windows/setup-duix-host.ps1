# Prepare the Windows PC that runs the Duix.Avatar Docker container for mywebtools.
# Run in an *elevated* PowerShell:  powershell -ExecutionPolicy Bypass -File setup-duix-host.ps1
param(
    [string]$Folder = "C:\AI\Duix\data\face2face\temp",
    [string]$ShareName = "duix_temp",
    [string]$Account = "$env:USERDOMAIN\$env:USERNAME",
    [int]$Port = 8383
)
$ErrorActionPreference = "Stop"

Write-Host "==> Folder $Folder"
New-Item -ItemType Directory -Force -Path $Folder | Out-Null

Write-Host "==> SMB share \\$env:COMPUTERNAME\$ShareName (read/write for $Account)"
if (-not (Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue)) {
    New-SmbShare -Name $ShareName -Path $Folder -FullAccess $Account | Out-Null
} else {
    Grant-SmbShareAccess -Name $ShareName -AccountName $Account -AccessRight Full -Force | Out-Null
}
$acl = Get-Acl $Folder
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule($Account, "Modify", "ContainerInherit,ObjectInherit", "None", "Allow")
$acl.SetAccessRule($rule); Set-Acl $Folder $acl

Write-Host "==> Firewall: allow TCP $Port (Duix API) and SMB on private networks"
if (-not (Get-NetFirewallRule -DisplayName "Duix Avatar API $Port" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "Duix Avatar API $Port" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private | Out-Null
}
Enable-NetFirewallRule -DisplayGroup "File and Printer Sharing" -ErrorAction SilentlyContinue

Write-Host "==> Docker container"
try {
    docker ps --filter "name=duix-avatar-gen-video" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    docker update --restart unless-stopped duix-avatar-gen-video | Out-Null
} catch {
    Write-Warning "docker is not running. Start Docker Desktop (Settings > General > 'Start Docker Desktop when you sign in')."
}

Write-Host "==> Wake-on-LAN (optional): MAC addresses of active adapters"
Get-NetAdapter | Where-Object Status -eq "Up" | Select-Object Name, MacAddress | Format-Table -AutoSize

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -ne "WellKnown" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "Done. On the board run:"
Write-Host "  sudo deploy/mount-duix-share.sh //$ip/$ShareName $env:USERNAME"
Write-Host "and in mywebtools Admin > Engine set API address http://${ip}:$Port"
