import os
from pathlib import Path

import certifi
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

load_dotenv(ROOT / ".env")

# The python.org macOS installer ships without a CA bundle, so HTTPS downloads made
# through urllib (which sportsdataverse's parquet loaders use) fail verification.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
