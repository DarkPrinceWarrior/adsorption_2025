# Adsorbent Inverse Design Framework (2025)

## Аннотация
Данный репозиторий реализует систему **обратного дизайна (Inverse Design)** пористых металл-органических каркасов (MOF). Система решает фундаментальную проблему материаловедения: поиск оптимальных условий синтеза для получения материала с заранее заданными структурно-энергетическими характеристиками (СЭХ).

В основе подхода лежит гибридная архитектура:
1.  **Deep Ensemble Forward Model:** Ансамбль из 5 градиентных бустингов (CatBoost) с per-target тюнингом гиперпараметров (Optuna) для предсказания свойств материала по условиям синтеза.
2.  **Conformal Prediction:** Калиброванные предиктивные интервалы с гарантированным покрытием (≥90%) на основе OOF-резидуалов и ensemble σ.
3.  **Multi-Objective Bayesian Optimization:** Алгоритм NSGA-II (Optuna) для построения Pareto-фронта оптимальных рецептов синтеза.

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
Система использует **True Deep Ensemble** + **Conformal Prediction**:

1.  **Deep Ensemble (5 моделей):** Обучаются на 100% данных с разными random seeds. Ensemble σ — мера эпистемической неопределённости.
2.  **Conformal Calibration:** OOF-резидуалы нормируются на ensemble σ для вычисления conformal quantile $q_\alpha$ с finite-sample correction (Vovk et al.). Предиктивный интервал: $\hat{y} \pm q_\alpha \cdot \sigma$.
3.  **Гарантированное покрытие:** Для α=0.10 (номинал 90%) фактическое покрытие: $E_0$ — 98.7%, $x_0$ — 98.7%, $S_{me}$ — 100%.

### Multi-Objective Bayesian Optimization
Вместо скаляризации целей используется **NSGA-II** (Optuna) для построения Pareto-фронта:
*   Каждый таргет — отдельная objective (minimize $|\hat{y} - y_{target}|$).
*   Физико-химические ограничения обрабатываются через `constraints_func` сэмплера.
*   Результат — множество Pareto-оптимальных рецептов + scalarized ranking для удобства выбора.

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

### Шаг 2: Обучение модели
```bash
PYTHONPATH=src python scripts/train_forward_model.py \
    --data data/SEC_SYN_with_features_enriched.csv \
    --validation-mode warn
```
*Результат:* В `artifacts/forward_models/`:
*   15 моделей CatBoost (5 ensemble members × 3 таргета)
*   `metrics.json` — OOF, production и interval-метрики
*   `uncertainty_calibrators.joblib` — CV-based conformal модели MAPIE
*   `predictions_*.csv` — OOF-предсказания + интервалы `y_lo/y_hi`

### Шаг 3: Валидация UQ
Скрипт строит **Rejection Plots** и проверяет **Interval Coverage**:
```bash
PYTHONPATH=src python scripts/validate_uncertainty.py
```
*Результат:* `artifacts/plots/uncertainty_rejection_plots.png`.

### Шаг 4: Поиск рецепта (Inverse Design)
Задайте желаемые СЭХ — target-oriented optimizer на BoFire domain models построит ранжированный список кандидатов.

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

---

## 4. Структура проекта

```
├── scripts/
│   ├── enrich_descriptors.py     # Шаг 0: Обогащение датасета (RDKit + коорд. химия)
│   ├── tune_hyperparams.py       # Шаг 1: Optuna HP tuning (5-fold CV)
│   ├── train_forward_model.py    # Шаг 2: Nested selection + production ensemble + MAPIE
│   ├── validate_uncertainty.py   # Шаг 3: UQ валидация (rejection plots + interval coverage)
│   ├── run_bofire_opt.py         # Шаг 4: Target-oriented inverse design
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
Каждый таргет имеет оптимальные гиперпараметры CatBoost (см. `config.py`):

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

### Прочее
*   **Physicochemical Constraints:** NSGA-II учитывает жёсткие ограничения через `constraints_func` (температурная монотонность, стехиометрия, точки кипения).
*   **Physics Penalties:** Sample weights увеличиваются для образцов с нарушениями ($a_0 = 28.86 \cdot W_0$, $E = E_0/3$).
*   **CatBoost:** Нативная работа с категориальными фичами без One-Hot кодирования.

---

## 6. Тестирование

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

**Покрытие:** 7 unit-тестов по валидации данных, инженерии признаков и расчёту молярных масс.
