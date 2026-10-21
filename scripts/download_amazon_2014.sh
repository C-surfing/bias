#!/usr/bin/env bash
# Resume Amazon 2014 5-core source archives; abort if any archive is incomplete.
set -euo pipefail
DEST="${1:-/data/cjh/rsi/project/datasets/amazon/raw}"
mkdir -p "$DEST"
cd "$DEST"
cat > amazon_2014_urls.txt <<'EOF'
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz
EOF
aria2c --input-file amazon_2014_urls.txt --continue=true \
  --max-connection-per-server=4 --split=4 --min-split-size=1M \
  --max-concurrent-downloads=3 --retry-wait=10 \
  --max-tries=4 --console-log-level=warn
for part in reviews_Beauty_5 meta_Beauty reviews_Sports_and_Outdoors_5 \
            meta_Sports_and_Outdoors reviews_CDs_and_Vinyl_5 \
            meta_CDs_and_Vinyl; do
    test ! -f "$part.json.gz.aria2"
    gzip -t "$part.json.gz"
done
sha256sum reviews_*.json.gz meta_*.json.gz > SHA256SUMS
echo "Verified all 6 source files in $DEST"
