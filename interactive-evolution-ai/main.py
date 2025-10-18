import sys
from pathlib import Path

# Ensure the project root is on sys.path for module imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))


if __name__ == "__main__":
    # Импорт внутри if __name__ для корректной работы multiprocessing на Windows
    from app.app_manager import AppManager
    AppManager().run()