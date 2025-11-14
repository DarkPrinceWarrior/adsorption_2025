# Adsorb Synthesis — конвейер инверсного проектирования сорбентов

Пайплайн обучает каскад моделей, которые по целевым адсорбционным характеристикам (W<sub>0</sub>, E<sub>0</sub>, S<sub>БЭТ</sub> и т. д.) восстанавливают параметры синтеза MOF-материала: металл, лиганд, массы прекурсоров, объём растворителя и температурные режимы. Все стадии используют общие словари дескрипторов, инженерные признаки и жёсткие физические ограничения (термодинамика, стехиометрия, температурные неравенства).

---

## Быстрый старт

```bash
# 1. Установить зависимости (Python ≥ 3.10)
python -m pip install -r requirements.txt

# 2. Обучить конвейер на подготовленном датасете
python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --output artifacts \
    --validation-mode strict    # строгая проверка данных (опционально)

# 3. Получить предсказания по новым адсорбционным измерениям
python scripts/predict_inverse_design.py \
    --model artifacts \
    --input tmp_inference_input.csv \
    --output tmp_predictions.csv \
    --targets-only \
    --validation-mode strict
```

- Все команды принимают `--validation-mode warn|strict` — в строгом режиме ошибки данных сразу приводят к остановке.
- CUDA не требуется: ансамбль (TabNet + CatBoost + XGBoost) корректно работает на CPU.

---

## Структура репозитория

```
├── scripts/                 # CLI
│   ├── train_inverse_design.py
│   ├── predict_inverse_design.py
│   └── tune_inverse_design.py
├── src/adsorb_synthesis/
│   ├── constants.py         # списки признаков, физические диапазоны, seeds
│   ├── data_processing.py   # загрузка, фичи, look-up таблицы, термодинамика
│   ├── modern_models.py     # ансамбль TabNet/CatBoost/XGBoost с физ. весами
│   ├── physics_losses.py    # валидатор, штрафы, проекторы
│   └── pipeline.py          # StageConfig, обучение, инференс
├── tests/                   # pytest
├── data/                    # исходные и подготовленные CSV
├── artifacts/               # модельные артефакты после обучения
└── README.md
```

---

## Данные и инженерные признаки

- **Сырые входы:** `data/SEC_SYN_with_features_DMFA_only_no_Y.csv` — очищенный датасет (только растворитель ДМФА, без металла Y).
- **Инженерные признаки (data_processing.py):**
  - `_ensure_adsorption_features` — проверяет базовые столбцы и достраивает вторичные величины (`Adsorption_Potential`, `S_BET_E`, `x0_W0`, и др.).
  - `add_salt_mass_features` — концентрации, лог-скейлы, стехиометрические отношения, температурные производные.
  - `add_thermodynamic_features` — пересчёт `K_eq ↔ ΔG` при фактической температуре синтеза. Оригинальные измерения не затираются: добавляются `Delta_G_equilibrium`, `K_equilibrium_from_delta_G`, `Delta_G_residual`, `K_equilibrium_ratio`.
  - `add_temperature_categories` — категоризация `Tsyn`, `Tdry`, `Treg`.
  - `build_lookup_tables` — словари дескрипторов металлов, лигандов и растворителей.

---

## Пайплайн и физические ограничения

### Стадии `default_stage_configs()`

1. `metal` — классификация металла по адсорбционному профилю.
2. `ligand` — выбор лиганда с учётом предсказанного металла.
3. `salt_mass` — регрессия log1p массы соли (StageConfig хранит `target_transform="log1p"` и `invert_target_to="m (соли), г"`).
4. `acid_mass`, `solvent_volume` — регрессии массы кислоты и объёма растворителя.
5. `tsyn_category`, `tdry_category`, `treg_category` — классификация температурных диапазонов.

Каждая стадия описывает зависимости (`depends_on`) и набор признаков; пайплайн автоматически применяет/обращает таргет-трансформы при обучении и предсказании.

### Физика

- **Валидаторы данных (`validate_SEH_data`, `validate_synthesis_data`)** — проверяют границы адсорбции, порядок температур, `T_syn` против точки кипения растворителя.
- **PhysicsConstraintEvaluator** — кроме диапазонов (`BoundConstraint`) использует:
  - `EqualityConstraint` (`a0 = 28.86·W0`);
  - `RatioConstraint` (`E/E0 = 1/3`);
  - `InequalityConstraint` (`Ws ≥ W0`).
- **`physics_violation_scores`** → веса выборки: теперь вычисляются векторно по подготовленному DataFrame, и TabNet получает те же веса, что CatBoost/XGBoost.
- **Проекции после инференса:** `project_thermodynamics` (делает `K_eq = exp(-ΔG/RT)`), `_project_stoichiometry` (отдельные цели для каждой пары металл-лиганд), `_enforce_temperature_limits` (числовая проверка + точки кипения).

---

## CLI-потоки

### Обучение

```
python scripts/train_inverse_design.py \
    --data data/SEC_SYN_with_features_DMFA_only_no_Y.csv \
    --output artifacts \
    --validation-mode strict
```

- После тренировки выводятся метрики по стадиям и `cv_mean/cv_std`.
- В `artifacts/` сохраняются пайплайны, lookup-таблицы и `pipeline_metadata.joblib`, который содержит StageConfig (включая `target_transform`, `physics_weight`, зависимости).

### Инференс

```
python scripts/predict_inverse_design.py \
    --model artifacts \
    --input tmp_inference_input.csv \
    --output tmp_predictions.csv \
    --targets-only
```

- Параметр `--targets-only` оставляет только финальные цели; без него выдаётся полный DataFrame с промежуточными признаками и физическими диагностическими колонками.
- Постобработка автоматически исправляет температурные и стехиометрические нарушения, а лог отображает корректировки (пример: «Adjusted Tdry_Category...»).

### Тюнинг

`python scripts/tune_inverse_design.py` — запускает Optuna-эксперимент для подбора `physics_weight`, `w_thermo`, `w_energy`, гиперпараметров ансамбля. Результаты сохраняются в `artifacts/tuning/`.

---

## Тесты

```
PYTHONPATH=. pytest
```

Набор покрывает:
- проверку инженерных и термодинамических колонок (`tests/test_feature_engineering.py`);
- стехиометрию, температурные коррекции и физические ограничения (`tests/test_physics_constraints.py`);
- корректность физ. весов в ансамбле (`tests/test_modern_models.py`);
- валидацию и словари молярных масс.

---

## Расширение системы

- **Новые признаки** добавляйте в `constants.py` (соответствующие списки) и в `data_processing.py` — так стадии автоматически увидят их через `StageConfig.feature_columns`.
- **Дополнительные ограничения** встраивайте в `PhysicsConstraintEvaluator`: реализованы структуры для диапазонов, равенств, отношений и неравенств.
- **Новые стадии** добавляйте в `default_stage_configs()`, описав зависимости и (при необходимости) физические столбцы (`physics_columns`), которые нужно передавать в модель.
- **Пакеты** — при необходимости добавляйте в `requirements.txt`, так как проект не использует `pyproject.toml`.

Любые изменения данных или моделей рекомендуется сопровождать: (1) повторным `train_inverse_design.py`, (2) `predict_inverse_design.py` на контрольной выборке и (3) прогоном `pytest`.
