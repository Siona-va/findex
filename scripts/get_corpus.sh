#!/usr/bin/env bash
# Downloads the corpus used in this project: the CPython documentation sources (*.rst).
# ~560 documents, ~13 MB of text. Result: data/pydocs/
set -euo pipefail
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse https://github.com/python/cpython.git "$tmp/cpython"
git -C "$tmp/cpython" sparse-checkout set Doc
mkdir -p data/pydocs
(cd "$tmp/cpython/Doc" && find . -name '*.rst' -exec cp --parents {} "$OLDPWD/data/pydocs/" \;)
rm -rf "$tmp"
echo "corpus ready: $(find data/pydocs -name '*.rst' | wc -l) files in data/pydocs/"
