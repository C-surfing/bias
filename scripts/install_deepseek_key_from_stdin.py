"""One-time credential installer. Reads secret via stdin, never logs or prints it."""
from pathlib import Path
import os
import sys

incoming = sys.stdin.buffer.read().decode("utf-8-sig").strip().lstrip("\ufeff")
if not incoming.isascii() or not incoming.startswith("sk-"):
    raise SystemExit("Invalid key encoding / format; nothing installed")
dest = Path("/data/cjh/rsi/.secrets/deepseek.key")
dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
os.chmod(dest.parent, 0o700)
tmp = dest.with_suffix(".tmp")
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    with os.fdopen(fd, "w", encoding="ascii") as file:
        file.write(incoming + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, dest)
finally:
    if tmp.exists():
        tmp.unlink()
print("SERVER_DEEPSEEK_CREDENTIAL_READY")
