"""Скрипт для запуска обучения на Kaggle/Colab с TensorNEAT.

Этот скрипт автоматически настраивает окружение для облачного обучения:
- Использует TensorNEAT с GPU-ускорением через JAX
- Автоматически определяет доступные устройства (CPU/GPU/TPU)
- Сохраняет результаты в доступное место

ВАЖНО: На Kaggle/Colab используется GPU версия JAX для максимального ускорения.
Для установки зависимостей выполните:
    pip install jax[cuda12] jaxlib  # Для CUDA 12
    pip install git+https://github.com/EMI-Group/tensorneat.git
"""
import os
import sys
from pathlib import Path

# Добавляем текущую директорию в PYTHONPATH
# Поддержка как скрипта, так и Jupyter Notebook
try:
    # Если запускается как скрипт
    project_root = Path(__file__).parent
except NameError:
    # Если запускается в Jupyter/Kaggle
    project_root = Path.cwd()

sys.path.insert(0, str(project_root))

# Импортируем TensorNEAT тренер вместо старого EvolutionManager
from app.core.tensorneat_trainer import create_trainer_from_settings
from app.utils.settings import Settings

def setup_for_cloud():
    """Настройка для облачного окружения с TensorNEAT."""
    print("=" * 60)
    print("🚀 НАСТРОЙКА ОБЛАЧНОГО ОБУЧЕНИЯ (TENSORNEAT)")
    print("=" * 60)

    # Проверяем доступность JAX и устройств
    try:
        import jax
        devices = jax.devices()
        print(f"✅ JAX установлен, версия: {jax.__version__}")
        print(f"📊 Доступно устройств JAX: {len(devices)}")
        for i, device in enumerate(devices):
            device_type = device.device_kind
            print(f"   [{i}] {device_type}: {device}")

        # Проверяем наличие GPU/TPU
        has_accelerator = any(d.device_kind in ['gpu', 'tpu'] for d in devices)
        if has_accelerator:
            print("🎉 Обнаружен GPU/TPU! Обучение будет использовать аппаратное ускорение.")
        else:
            print("⚠️  GPU/TPU не обнаружен. Обучение будет на CPU (медленнее).")

    except ImportError:
        print("❌ JAX не установлен! Установите зависимости:")
        print("   pip install jax[cuda12] jaxlib  # Для CUDA 12")
        print("   pip install git+https://github.com/EMI-Group/tensorneat.git")
        sys.exit(1)

    # Определяем количество доступных ядер (для информации)
    cpu_count = os.cpu_count() or 1
    print(f"\n💻 Доступно CPU ядер: {cpu_count}")

    # Определяем корневую директорию проекта
    try:
        root = Path(__file__).parent
    except NameError:
        root = Path.cwd()

    # Загружаем настройки
    config_path = root / "config" / "settings.ini"
    settings = Settings(config_path)

    # Выводим текущие настройки
    print("\n📋 Параметры обучения:")
    print(f"   - Популяция: {settings.get_int('Simulation', 'population_size')}")
    print(f"   - Поколения: {settings.get_int('Simulation', 'generations')}")
    print(f"   - Размер поля: {settings.get_int('Field', 'field_size')}")
    print(f"   - Еда: {settings.get_int('Simulation', 'food_quantity')}")
    print(f"   - Респавн еды: {settings.get_bool('Simulation', 'food_respawn')}")

    print("\n⚡ ПРЕИМУЩЕСТВА TENSORNEAT:")
    print("   - GPU-ускорение через JAX (до 500x быстрее)")
    print("   - Автоматическая векторизация популяции")
    print("   - JIT-компиляция для максимальной производительности")

    print("=" * 60)
    print()

    return settings

def main():
    """Главная функция для облачного обучения с TensorNEAT."""
    # Настраиваем окружение
    settings = setup_for_cloud()

    # Определяем корневую директорию проекта
    try:
        root = Path(__file__).parent
    except NameError:
        root = Path.cwd()

    # Создаём директорию для сохранения результатов
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("🎯 Запуск обучения с TensorNEAT...\n")

    try:
        # Создаём TensorNEAT тренер
        trainer = create_trainer_from_settings(settings, seed=42)

        # Запускаем обучение
        state, best = trainer.run_training(verbose=True)

        # Сохраняем лучший геном
        if state is not None:
            save_path = trainer.save_best_genome(state, output_dir=checkpoints_dir)
            print(f"\n💾 Результаты сохранены в: {save_path}")

        print("\n✅ Обучение успешно завершено!")
        print(f"🏆 Лучший фитнес: {best}")

    except KeyboardInterrupt:
        print("\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()