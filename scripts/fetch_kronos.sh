#!/bin/zsh
# Fetch Kronos (MIT, https://github.com/shiyu-coder/Kronos) into vendor/github/Kronos at the
# commit this project was developed against. Model weights download from Hugging Face on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=67b630e67f6a18c9e9be918d9b4337c960db1e9a
DEST=vendor/github/Kronos
[[ -f ${DEST}/model/kronos.py ]] && { echo "Kronos already in ${DEST}"; exit 0; }
mkdir -p ${DEST}
curl -fsSL "https://codeload.github.com/shiyu-coder/Kronos/tar.gz/${REV}" | tar -xzf - --strip-components=1 -C ${DEST}
echo "Kronos ${REV:0:7} → ${DEST}"
