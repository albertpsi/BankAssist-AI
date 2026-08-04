"""Dev launcher: chdir to project root, then exec Streamlit on the Lab 4/5 Agentic
Assistant page, so relative paths (.env) resolve correctly."""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

if __name__ == "__main__":
    port = os.environ.get("PORT", "8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "src" / "bankassist" / "ui" / "agentic_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            port,
        ],
        check=True,
    )
