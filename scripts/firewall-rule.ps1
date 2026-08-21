# Windows Firewall rule for Daryl-Memories MCP server
# Run as Administrator to allow connections from other machines on port 8080

param(
    [string[]]$ClientIPs = @(),
    [switch]$Remove
)

$RuleName = "Daryl-Memories MCP (TCP 8080)"

if ($Remove) {
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    Write-Host "Removed firewall rule: $RuleName"
    exit 0
}

if ($ClientIPs.Count -eq 0) {
    Write-Host "Usage:"
    Write-Host "  .\scripts\firewall-rule.ps1 -ClientIPs '192.168.1.100','192.168.1.101'"
    Write-Host "  .\scripts\firewall-rule.ps1 -Remove"
    exit 1
}

# Remove existing rule if present
Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue

# Create new rule: allow inbound TCP 8080 from specific IPs only
New-NetFirewallRule `
    -DisplayName $RuleName `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8080 `
    -Action Allow `
    -RemoteAddress $ClientIPs `
    -Description "Allow Daryl-Memories MCP connections from agent machines" `
    -Profile Private

Write-Host "Firewall rule created: $RuleName"
Write-Host "Allowed IPs: $($ClientIPs -join ', ')"
Write-Host ""
Write-Host "Verify with: Get-NetFirewallRule -DisplayName '$RuleName' | Format-List"
