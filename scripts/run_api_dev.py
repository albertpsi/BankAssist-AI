"""Dev launcher: ensures the working directory is the project root before uvicorn
starts, so relative paths in Settings (.env, data/policies, data/banking.db) resolve
correctly regardless of the caller's own cwd."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bankassist.api.app:create_app", factory=True, port=8000, host="127.0.0.1")
