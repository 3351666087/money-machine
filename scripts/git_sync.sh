#!/bin/zsh
# Commit everything that is not ignored and push to the public GitHub repo.
# The pre-commit hook (scripts/check_secrets.sh) refuses secrets, public IPs and personal e-mails.
#   scripts/git_sync.sh "short summary"
set -euo pipefail
cd "$(dirname "$0")/.."
MSG=${1:?usage: scripts/git_sync.sh "summary"}
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "${MSG}" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q
git log --oneline -1
