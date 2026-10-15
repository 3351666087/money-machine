#!/bin/zsh
# Pre-commit guard for the public repo: refuse to commit staged content that looks like a secret,
# a public IP address or a personal e-mail address. Install: ln -s ../../scripts/check_secrets.sh .git/hooks/pre-commit
set -uo pipefail
files=$(git diff --cached --name-only --diff-filter=ACM)
[[ -z ${files} ]] && exit 0
bad=0
check() {  # label, extended regex
  local hits
  hits=$(echo "${files}" | tr '\n' '\0' | xargs -0 grep -nIE "$2" 2>/dev/null | grep -vE "$3")
  if [[ -n ${hits} ]]; then echo "BLOCKED ($1):"; echo "${hits}" | head -5; bad=1; fi
}
check "token / key" '(gho_|ghp_|github_pat_|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)' 'sk-xxxx'
check "filled secret in env syntax" '^(MM_[A-Z_]*(TOKEN|KEY|PASSWORD|WEBHOOK))=[^[:space:]]+' 'example\.com|sk-xxxx'
# public IPv4 addresses (private, loopback and documentation ranges are fine)
ips=$(echo "${files}" | tr '\n' '\0' | xargs -0 grep -HnoIE '(^|[^0-9.])([0-9]{1,3}\.){3}[0-9]{1,3}([^0-9.]|$)' 2>/dev/null \
  | python3 -c '
import ipaddress, re, sys
for line in sys.stdin:
    parts = line.rstrip("\n").split(":", 2)
    if len(parts) < 3:
        print("unparsed:" + line.strip()); continue
    f, n, text = parts
    for ip in re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", text):
        try:
            if ipaddress.ip_address(ip).is_global:
                print(f"{f}:{n}: {ip}")
        except ValueError:
            pass
')
if [[ ${pipestatus[-1]:-0} != 0 || -n ${ips} ]]; then echo "BLOCKED (public IP address or scan error):"; echo "${ips}" | head -5; bad=1; fi
check "personal e-mail" '[A-Za-z0-9._%+-]+@(gmail|qq|163|126|outlook|hotmail|icloud|yahoo)\.com' 'example|noreply'
[[ ${bad} == 1 ]] && { echo "commit refused by scripts/check_secrets.sh"; exit 1; }
exit 0
