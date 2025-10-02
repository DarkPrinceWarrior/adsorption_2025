# Инверсионный дизайн адсорбентов

Репозиторий содержит полный цикл обратного проектирования МОF-адсорбентов: подготовку данных СЭХ, обучение последовательности моделей и инференс технологических параметров (металл, лиганд, растворитель, массы реагентов, объём растворителя и температурные режимы). Используется стек Python 3.10 + HistGradientBoosting (scikit-learn ≥ 1.7.2), а процессы завернуты в CLI-скрипты.

## 1. Данные

### 1.1 Исходные файлы
- `data/SEC_SYN_new.csv` — сырые записи синтезов с технологическими параметрами.
- `data/SEC_SYN_with_features.csv` — очищенный и дополненный датасет для обучения/инференса.

### 1.2 Группы признаков
Перечни признаков вынесены в `src/adsorb_synthesis/constants.py`.

- **ADSORPTION_FEATURES**: `W0, см3/г`, `E0, кДж/моль`, `х0, нм`, `а0, ммоль/г`, `E, кДж/моль`, `SБЭТ, м2/г`, `Ws, см3/г`, `Sme, м2/г`, `Wme, см3/г`, `Adsorption_Potential`, `Capacity_Density`, `SurfaceArea_MicroVol_Ratio`, `Adsorption_Energy_Ratio`, `S_BET_E`, `x0_W0`, `K_equilibrium`, `Delta_G`, `B_micropore`.
- **METAL_DESCRIPTOR_FEATURES**: молярная масса, средний ионный радиус, электроотрицательность, `Молярка_соли` и др.
- **LIGAND_DESCRIPTOR_FEATURES**: число карбоксильных/аминогрупп, ароматические кольца, атомный состав, дескрипторы Lipinski/TPSA, `Молярка_кислоты`.
- **SOLVENT_DESCRIPTOR_FEATURES**: молярная масса, LogP, количество доноров/акцепторов водородных связей.
- **PROCESS_CONTEXT_FEATURES**: вспомогательные параметры процесса (например, `Mix_solv_ratio`).

### 1.3 Категории температур
К температурным столбцам добавляются категориальные версии (`Tsyn_Category`, `Tdry_Category`, `Treg_Category`).

| Процесс      | Низкая       | Средняя       | Высокая       |
|--------------|--------------|---------------|---------------|
| Синтез       | < 115 °C     | 115–135 °C    | > 135 °C      |
| Сушка        | < 115 °C     | 115–135 °C    | > 135 °C      |
| Регенерация  | < 155 °C     | 155–265 °C    | > 265 °C      |

Разметка выполняется в `add_temperature_categories`.

## 2. Архитектура проекта

```
adsorb_synthesis/
├── data/
├── scripts/
│   ├── train_inverse_design.py
│   └── predict_inverse_design.py
├── src/adsorb_synthesis/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_processing.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

- `constants.py` — списки признаков, температурные диапазоны, случайные константы.
- `data_processing.py` — загрузка CSV, пересчёт инженерных признаков, построение lookup-таблиц.
- `pipeline.py` — конфигурация стадий, обучение/инференс, сериализация.
- `scripts/` — командные инструменты для обучения и инференса.

## 3. Подготовка данных
1. `load_dataset` читает CSV, проверяет обязательные столбцы, формирует температурные категории.
2. `_ensure_adsorption_features` пересчитывает производные показатели адсорбции, если их нет в файле.
3. `build_lookup_tables` создаёт справочники дескрипторов для каждого металла, лиганда и растворителя.

## 4. Модели и последовательность стадий

### 4.1 Алгоритмы
- `HistGradientBoostingClassifier` и `HistGradientBoostingRegressor` с параметрами: `max_iter=500`, `learning_rate=0.05`, `l2_regularization=0.1`. Для классификации используется `class_weight="balanced"`.
- Регрессионные таргеты очищаются от выбросов `IsolationForest(contamination=0.05)`.

### 4.2 Стадии `default_stage_configs`
1. **metal** — металл по адсорбционным признакам.
2. **ligand** — лиганд с учётом адсорбции и дескрипторов металла.
3. **solvent** — растворитель (использует `Металл` и `Лиганд`).
4. **salt_mass** — масса соли `m (соли), г`.
5. **acid_mass** — масса кислоты `m(кис-ты), г`.
6. **solvent_volume** — объём растворителя `Vсин. (р-ля), мл`.
7. **tsyn_category** — категория температуры синтеза.
8. **tdry_category** — категория температуры сушки (зависит от `Tsyn_Category`).
9. **treg_category** — категория температуры регенерации (учитывает `Tsyn_Category` и `Tdry_Category`).

Каждая стадия опирается только на уже предсказанные или исходные признаки, что предотвращает утечки.

### 4.3 Предобработка
### 4.4 Используемые признаки по таргетам
Для прозрачности перечислены признаки, подаваемые на каждую модель (группы раскрываются через списки в constants.py).
- **Металл** — только ADSORPTION_FEATURES.
- **Лиганд** — ADSORPTION_FEATURES + METAL_DESCRIPTOR_FEATURES + промежуточный столбец Металл.
- **Растворитель** — ADSORPTION_FEATURES + METAL_DESCRIPTOR_FEATURES + LIGAND_DESCRIPTOR_FEATURES + столбцы Металл, Лиганд.
- **m (соли), г** — все признаки для растворителя + SOLVENT_DESCRIPTOR_FEATURES + PROCESS_CONTEXT_FEATURES + Растворитель.
- **m(кис-ты), г** — признаки предыдущего шага + таргет m (соли), г.
- **Vсин. (р-ля), мл** — признаки предыдущего шага + таргет m(кис-ты), г.
- **Tsyn_Category** — признаки предыдущего шага + таргет Vсин. (р-ля), мл.
- **Tdry_Category** — признаки для Tsyn_Category + предсказанный Tsyn_Category.
- **Treg_Category** — признаки для Tdry_Category + предсказанный Tdry_Category.
Каждый последующий этап использует результат предыдущих моделей в виде новых колонок, которые подставляются автоматически в процессе инференса.
`_build_pipeline` делит признаки на числовые/категориальные, выполняет `SimpleImputer` и `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, после чего модель принимает плотные матрицы.

## 5. Обучение

```bash
python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features.csv \
    --output artifacts
```

Команда выполняет:
1. Загрузку данных, достраивание признаков, создание температурных категорий.
2. Построение lookup-таблиц.
3. Обучение всех стадий по очереди (train/test split, кросс-валидация, outlier filtering для регрессий).
4. Сохранение метрик: для классификации `accuracy`, `balanced_accuracy`, `f1_macro`; для регрессии `R²`, `RMSE`, `MAE` + `cv_mean`/`cv_std`.
5. Сериализацию пайплайнов (`*_pipeline.joblib`), lookup-таблиц и метаданных (`pipeline_metadata.joblib`).

## 6. Инференс

### 6.1 Подготовка входного CSV
Нужны все признаки из `ADSORPTION_FEATURES` (желательно добавить `Mix_solv_ratio`). Пример подготовки тестового файла:

```bash
python -X utf8 -c "import pandas as pd; df = pd.read_csv('data/SEC_SYN_with_features.csv'); \
    cols = ['W0, см3/г','E0, кДж/моль','х0, нм','а0, ммоль/г','E, кДж/моль','SБЭТ, м2/г',\
            'Ws, см3/г','Sme, м2/г','Wme, см3/г','Adsorption_Potential','Capacity_Density',\
            'SurfaceArea_MicroVol_Ratio','Adsorption_Energy_Ratio','S_BET_E','x0_W0',\
            'K_equilibrium','Delta_G','B_micropore','Mix_solv_ratio']; \
    df.loc[:4, cols].to_csv('tmp_inference_input.csv', index=False)"
```

### 6.2 Запуск предсказаний

```bash
python scripts/predict_inverse_design.py \
    --model artifacts \
    --input tmp_inference_input.csv \
    --targets-only
```

На каждом шаге пайплайн автоматически достраивает дескрипторы через `_augment_with_lookup_descriptors`, проверяет наличие нужных признаков и выдаёт финальные столбцы: `Металл`, `Лиганд`, `Растворитель`, `m (соли), г`, `m(кис-ты), г`, `Vсин. (р-ля), мл`, `Tsyn_Category`, `Tdry_Category`, `Treg_Category`. Флаг `--targets-only` убирает исходные признаки из вывода.

## 7. Структура артефактов

```
artifacts/
├── metal_pipeline.joblib
├── ligand_pipeline.joblib
├── solvent_pipeline.joblib
├── salt_mass_pipeline.joblib
├── acid_mass_pipeline.joblib
├── solvent_volume_pipeline.joblib
├── tsyn_category_pipeline.joblib
├── tdry_category_pipeline.joblib
├── treg_category_pipeline.joblib
├── lookup_metal.joblib
├── lookup_ligand.joblib
├── lookup_solvent.joblib
└── pipeline_metadata.joblib
```

`pipeline_metadata.joblib` хранит информацию о стадиях, признаках, метриках и относительных путях к моделям; при `InverseDesignPipeline.load` конфигурация восстанавливается автоматически.

## 8. Окружение

```bash
python -m pip install -r requirements.txt
```

Файл содержит `numpy`, `pandas`, `scikit-learn`, `joblib`. Для жёсткой фиксации окружения рекомендуются `pyproject.toml`/`poetry.lock` или `conda`.

## 9. Контроль качества и воспроизводимость
- Фиксированный `RANDOM_SEED=42` обеспечивает повторяемость сплитов.
- Кросс-валидация динамически подбирает число фолдов (StratifiedKFold → KFold при нехватке наблюдений).
- При инференсе отсутствие обязательных колонок приводит к понятному исключению с перечислением недостающих признаков.

## 10. Дальнейшее развитие
1. Добавить unit-тесты для `default_stage_configs`, `_augment_with_lookup_descriptors`, CLI.
2. Зафиксировать окружение в `pyproject.toml` / `poetry.lock` либо `environment.yml`.
3. Провести гипертюнинг (Optuna) и протестировать CatBoost/LightGBM для стадий с дисбалансом классов.
4. Интегрировать интерпретацию моделей (SHAP, permutation importance).

## 11. FAQ
- **Можно ли обучить на своих данных?** Да, при сохранении структуры колонок и корректной генерации дескрипторов.
- **Как добавить новую стадию?** Расширить `default_stage_configs`, определить `depends_on`, при необходимости обновить CLI и сериализацию.
- **Что делать при ошибке нехватки признаков во время инференса?** Добавить перечисленные столбцы во входной CSV или расширить функцию подготовки признаков.

## 12. Быстрый старт
1. Установить зависимости: `python -m pip install -r requirements.txt`.
2. Обучить пайплайн: `python scripts/train_inverse_design.py --data data/SEC_SYN_with_features.csv --output artifacts`.
3. Подготовить файл с признаками: см. пример выше (`tmp_inference_input.csv`).
4. Получить предсказания: `python scripts/predict_inverse_design.py --model artifacts --input tmp_inference_input.csv --targets-only`.

Система готова к использованию и соответствует требованиям на 1 октября 2025 года.

