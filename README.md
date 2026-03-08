# Adsorbent Inverse Design Framework (2026)

## Аннотация
Данный репозиторий реализует систему **обратного дизайна (Inverse Design)** пористых металл-органических каркасов (MOF). Система решает фундаментальную проблему материаловедения: поиск оптимальных условий синтеза для получения материала с заранее заданными структурно-энергетическими характеристиками (СЭХ).

В основе подхода лежит production-core pipeline с отдельными research challengers:
1.  **Production Forward Model:** `CatBoost + MAPIE` с fold-local feature selection и OOF/production артефактами.
2.  **Production Inverse Design:** practical optimizer на `BoFire`-based domain/constraint layer.
3.  **Wave 2 Research Branches:** `TabPFN`, `BoTorch`, `BayBE`, `direct inverse`, единый benchmark и comparative report.

---

## Рекомендованный стек

По состоянию на **8 марта 2026** после полного `wave2`-suite рекомендуемый рабочий стек такой:

*   **Forward backend:** `CatBoost`
*   **Uncertainty Quantification:** `MAPIE` intervals
*   **Основной inverse backend:** `run_bofire_opt.py`
*   **Research optimizer:** `run_botorch_mobo.py`
*   **Campaign backend:** `run_baybe_campaign.py`
*   **Direct inverse:** только benchmark baseline, не основной workflow

Подробное решение зафиксировано в [wave2_decision.md](/home/ruslan_safaev/adsorb_synth/adsorb_synthesis/wave2_decision.md).

---

## 1. Методология

### Прямая задача (Forward Problem)
Мы моделируем функцию $f: \text{Synthesis} \to \text{Properties}$.
*   **Входные данные ($X$):**
    *   *Химические реагенты:* Тип металла, тип лиганда, растворитель.
    *   *Физико-химические дескрипторы металла:* Ионный радиус, степень окисления, электронное сродство, эффект Яна-Теллера.
    *   *Дескрипторы лиганда:* Молекулярная масса, число карбоксильных групп.
    *   *Параметры процесса:* Температуры синтеза, сушки и регенерации ($T_{syn}, T_{dry}, T_{reg}$), стехиометрические соотношения ($R_{molar}$), концентрации ($C_{metal}$, молярности).
    *   *Физико-химические:* `n_water_hidden`, `Supersaturation_Index`, `Reactor_Loading_g_mL` и др.
*   **Целевые переменные ($Y$):**
    *   $E_0$ [кДж/моль] — Характеристическая энергия адсорбции.
    *   $x_0$ [нм] — Характеристическая полуширина пор.
    *   $S_{me}$ [м²/г] — Удельная поверхность мезопор.

### Оценка неопределенности (Uncertainty Quantification)
Production-ветка использует **CV-based conformal intervals через MAPIE**:

1.  **Fold-local CatBoost pipeline:** outer-CV без leakage из feature selection.
2.  **Cross-Conformal Regression:** интервалы `y_lo/y_hi` строятся через `MAPIE` по той же CV-схеме.
3.  **Единые артефакты:** `predictions_<target>.csv` содержат `y_actual`, `y_oof`, `y_prod_mean`, `y_lo`, `y_hi`, `interval_width`.
4.  **Validation layer:** `validate_uncertainty.py` проверяет empirical coverage и rejection curve по `interval_width`.

### Multi-Objective Bayesian Optimization
Production inverse stage теперь строится не вокруг legacy `NSGA-II`, а вокруг **target-oriented practical optimization**:
*   Основной workflow: `run_bofire_opt.py`
*   Явные constraints: temperature order, boiling point, stoichiometry, bounds по `E0`
*   Выход: shortlist кандидатов с `score`, `feasible`, prediction intervals и diagnostics
*   Legacy `run_bayes_opt.py` сохранён как historical baseline

---

## 2. Установка и Настройка

Требуется Python 3.10+. Зависимости зафиксированы в `requirements.txt`.

```bash
# 1. Создание виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Запуск тестов
PYTHONPATH=src python -m pytest tests/ -v
```

---

## 3. Руководство пользователя (Workflow)

### Шаг 0: Обогащение дескрипторов
Если у вас только базовый датасет `data/SEC_SYN_with_features.csv`, сначала обогатите его:
```bash
PYTHONPATH=src python scripts/enrich_descriptors.py \
    --input data/SEC_SYN_with_features.csv \
    --output data/SEC_SYN_with_features_enriched.csv
```

### Шаг 1: Тюнинг гиперпараметров (опционально)
Автоматический подбор гиперпараметров CatBoost для каждого таргета через Optuna с fold-local feature selection:
```bash
PYTHONPATH=src python scripts/tune_hyperparams.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --trials 80
```
*Результат:* `artifacts/best_hyperparams.json` + snippet для `config.py`.

### Шаг 2: Обучение production forward-модели
```bash
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --backend catboost \
    --validation-mode warn
```
*Результат:* В `artifacts/forward_models/`:
*   15 production-моделей CatBoost (5 ensemble members × 3 таргета)
*   `metrics.json` — OOF, production и interval-метрики
*   `uncertainty_calibrators.joblib` — CV-based conformal модели MAPIE
*   `predictions_*.csv` — OOF-предсказания + интервалы `y_lo/y_hi`

### Шаг 3: Валидация UQ
Скрипт строит **Rejection Plots** и проверяет **Interval Coverage**:
```bash
PYTHONPATH=src python scripts/validate_uncertainty.py
```
*Результат:* `artifacts/plots/uncertainty_rejection_plots.png`.

### Шаг 4: Production inverse design
Задайте желаемые СЭХ — target-oriented optimizer на `BoFire` domain models построит ранжированный shortlist кандидатов.

```bash
PYTHONPATH=src python scripts/run_bofire_opt.py \
    --E0 15.0 \
    --x0 0.5 \
    --Sme 100.0 \
    --trials 300 \
    --shortlist-size 12 \
    --output artifacts/predictions_bofire.csv
```

**Аргументы:**
*   `--E0`, `--x0`, `--Sme`: целевые значения свойств.
*   `--trials`: бюджет поиска, то есть сколько кандидатов optimizer просмотрит внутри.
*   `--shortlist-size`: сколько diverse-кандидатов сохранить в итоговый CSV.
*   `--output`: путь к итоговому shortlist CSV.
*   `--all-output`: опциональный путь для сохранения полного пула просмотренных кандидатов.

*Результат:* CSV с небольшим diverse shortlist кандидатов, включая условия синтеза, предсказанные свойства, интервалы `Pred_*_lo/hi`, `feasible`, `constraint_reasons`, итоговый `score` и `search_rank` исходного поиска.

### Шаг 5: Wave 2 comparative workflow
Если нужно сравнить runnable research-ветки второй волны:

```bash
PYTHONPATH=src python scripts/run_wave2_suite.py \
    --mode full \
    --tabpfn-only \
    --catboost-models artifacts/forward_models \
    --bofire-shortlist artifacts/predictions_bofire_shortlist.csv \
    --bofire-pool artifacts/predictions_bofire_all.csv \
    --botorch-shortlist artifacts/predictions_botorch.csv \
    --botorch-pool artifacts/predictions_botorch_all.csv \
    --data data/SEC_SYN_with_features_enriched.csv \
    --E0 15 \
    --x0 0.5 \
    --Sme 100 \
    --run-dir artifacts/wave2_runs/selection_run
```

Скрипт:
*   досчитает `TabPFN`, `BayBE`, `direct inverse`, если артефактов нет;
*   соберёт `benchmark_wave2.py`;
*   создаст comparative report через `generate_wave2_report.py`.

---

## 4. Структура проекта

```
├── scripts/
│   ├── enrich_descriptors.py     # Шаг 0: Обогащение датасета (RDKit + коорд. химия)
│   ├── tune_hyperparams.py       # Шаг 1: Optuna HP tuning (5-fold CV)
│   ├── train_forward_model.py    # Шаг 2: Nested selection + production ensemble + MAPIE
│   ├── validate_uncertainty.py   # Шаг 3: UQ валидация (rejection plots + interval coverage)
│   ├── run_bofire_opt.py         # Шаг 4: Production inverse design
│   ├── run_botorch_mobo.py       # Research inverse optimizer
│   ├── run_baybe_campaign.py     # Campaign-oriented optimizer scaffold
│   ├── train_inverse_direct.py   # Direct inverse benchmark baseline
│   ├── benchmark_wave2.py        # Comparative benchmark layer
│   ├── generate_wave2_report.py  # Comparative report layer
│   ├── run_wave2_suite.py        # Orchestrator for runnable wave 2
│   ├── run_bayes_opt.py          # Legacy baseline (Optuna NSGA-II)
│   └── generate_paper_figures.py # Фигуры для статьи/отчета
├── src/
│   └── adsorb_synthesis/
│       ├── config.py             # Конфигурация моделей (per-target tuned HP)
│       ├── constants.py          # Справочники (Molar Masses, Features, Targets)
│       ├── data_processing.py    # Генерация дескрипторов (inplace=True/False)
│       ├── data_validation.py    # Валидация физических ограничений
│       ├── feature_selection.py  # Feature Selection (VIF, корреляции, domain knowledge)
│       └── physics_losses.py     # Физические constraints и penalties
├── tests/                        # Unit-тесты (7 тестов)
├── data/                         # Экспериментальные датасеты (380 образцов)
└── artifacts/                    # Модели, метрики, графики, результаты BO
```

## 5. Особенности реализации

### Feature Engineering Pipeline
Для каждого таргета автоматически:
1. **Domain-driven curation:** Экспертный список keep/drop фичей (дедупликация обратных признаков: `Vsyn_m` ↔ `C_metal`, `R_mass` ↔ `R_molar`)
2. **Удаление мультиколлинеарности:** Фичи с |r| > 0.85 и VIF > 10 убираются итеративно
3. **Hard keep policy:** curated physics features принудительно сохраняются, если доступны
4. **Permutation Importance:** Финальный отбор топ-15 гибких фич
5. **No data leakage:** feature selection выполняется отдельно внутри каждого outer fold

> **Примечание:** RDKit-дескрипторы лиганда (3D geometry, 2D topological) исключены из модели — при 4 уникальных лигандах они вырождаются в lookup-таблицу из 4 строк. Категориальный признак `Лиганд` + `carboxyl_groups` + `molecular_weight` достаточны.

### Физико-химические дескрипторы
*   **Металл:** `ionic_radius_pm`, `electron_affinity_kj`, `oxidation_state`, `Jahn_Teller_Active`
*   **Лиганд:** `carboxyl_groups`, `molecular_weight`
*   **Взаимодействие:** `Metal_Ligand_Size_Ratio`
*   **Физико-химические:** `n_water_hidden`, `Supersaturation_Index`, `Molarity_Metal`, `Reactor_Loading_g_mL`

### Per-Target Hyperparameter Tuning
Каждый таргет имеет production-гиперпараметры CatBoost (см. `config.py`):

| Таргет | iterations | learning_rate | depth | l2_leaf_reg |
|--------|-----------|---------------|-------|-------------|
| $E_0$  | 1700      | 0.034         | 8     | 2.2         |
| $x_0$  | 1600      | 0.064         | 7     | 1.85        |
| $S_{me}$ | 2000    | 0.047         | 6     | 0.11        |

### Валидация данных
`validate_synthesis_data` поддерживает режимы `warn`/`strict`. Текущий датасет содержит строки с нарушением температурного порядка, точек кипения и стехиометрии — используйте `warn` (по умолчанию).

### Data Processing Safety
Все функции мутации DataFrame поддерживают параметр `inplace`:
```python
df_new = add_salt_mass_features(df, inplace=False)  # безопасная копия
add_salt_mass_features(df)  # inplace=True (по умолчанию)
```

### Wave 2 Status
*   **CatBoost:** текущий production default.
*   **TabPFN:** challenger backend; на текущем датасете слабее CatBoost, особенно по `Sme`.
*   **BoFire:** лучший inverse backend на текущем benchmark run.
*   **BoTorch:** рабочий research backend, но пока уступает BoFire по `best_score_pool`.
*   **BayBE:** рабочий campaign backend для low-data loop, но не лучший production optimizer.
*   **Direct inverse:** полезный benchmark, но не рекомендуется как основной способ подбора рецепта.

---

## 6. Тестирование

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

**Покрытие:** unit-тесты по валидации данных, инженерии признаков и расчёту молярных масс + smoke/regression проверки через runnable scripts.
