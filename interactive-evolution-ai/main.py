import sys
from pathlib import Path

# Ensure the project root is on sys.path for module imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from app.app_manager import AppManager  # noqa: E402


if __name__ == "__main__":
    AppManager().run()