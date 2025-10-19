"""Скрипт для запуска обучения на Kaggle/Colab.

Этот скрипт автоматически настраивает окружение для облачного обучения:
- Отключает визуализацию
- Настраивает количество workers под доступные ядра
- Сохраняет результаты в доступное место
"""
import os
import sys
from pathlib import Path

# Добавляем текущую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app.app_manager import AppManager
from app.utils.settings import Settings

def setup_for_cloud():
    """Настройка для облачного окружения."""
    print("=" * 60)
    print("🚀 НАСТРОЙКА ОБЛАЧНОГО ОБУЧЕНИЯ")
    print("=" * 60)

    # Определяем количество доступных ядер
    cpu_count = os.cpu_count() or 1
    print(f"📊 Доступно CPU ядер: {cpu_count}")

    # Настраиваем workers (оставляем 1 ядро для системы)
    recommended_workers = max(1, cpu_count - 1)
    print(f"⚙️  Рекомендуемое количество workers: {recommended_workers}")

    # Загружаем и обновляем настройки
    config_path = Path(__file__).parent / "config" / "settings.ini"
    settings = Settings(config_path)

    # Обновляем workers
    current_workers = settings.get_int("Simulation", "workers")
    if current_workers != recommended_workers:
        settings.set_int("Simulation", "workers", recommended_workers)
        print(f"✅ Workers обновлены: {current_workers} → {recommended_workers}")
    else:
        print(f"✅ Workers уже оптимизированы: {recommended_workers}")

    # Выводим текущие настройки
    print("\n📋 Параметры обучения:")
    print(f"   - Популяция: {settings.get_int('Simulation', 'population_size')}")
    print(f"   - Поколения: {settings.get_int('Simulation', 'generations')}")
    print(f"   - Еда: {settings.get_int('Simulation', 'food_quantity')}")
    print(f"   - Респавн еды: {settings.get_bool('Simulation', 'food_respawn')}")
    print(f"   - Матчей на геном: {settings.get_int('Simulation', 'matches_per_genome')}")
    print(f"   - Workers: {recommended_workers}")
    print("=" * 60)
    print()

    return settings

def main():
    """Главная функция для облачного обучения."""
    # Настраиваем окружение
    settings = setup_for_cloud()

    # Создаём AppManager БЕЗ визуализации
    print("🎯 Запуск обучения (без визуализации)...\n")
    app = AppManager(visualize=False)

    try:
        # Запускаем эволюцию
        app.run_evolution()
        print("\n✅ Обучение успешно завершено!")

    except KeyboardInterrupt:
        print("\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        raise

if __name__ == "__main__":
    main()
