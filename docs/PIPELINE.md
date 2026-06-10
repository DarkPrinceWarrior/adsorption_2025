# Сквозной пайплайн adsorb_synthesis — полное техническое описание

Версия: 2026-06-10. Ветка: `Bayesian-Optimization`.
Документ описывает весь сквозной конвейер обратного дизайна пористых металл-органических
каркасов (MOF): от сырых Excel-файлов измерений до предсказания трёх структурно-энергетических
характеристик (СЭХ) и обратного подбора условий синтеза.

Сопутствующие документы: [README.md](../README.md) (пользовательский workflow),
[wave2_decision.md](../wave2_decision.md) (выбор стека), [pipeline_audit_2026-06-09.md](pipeline_audit_2026-06-09.md)
(аудит недостатков), [pipeline_audit_fixes_2026-06-09.md](pipeline_audit_fixes_2026-06-09.md)
(внедрённые исправления), [data/DATA_DICTIONARY.md](../data/DATA_DICTIONARY.md) (словарь данных + checksum).

---

## 1. Постановка задачи

Решается задача **обратного дизайна (inverse design)** материалов: найти условия синтеза MOF,
дающие материал с заранее заданными СЭХ. Декомпозируется на две стадии.

**Прямая задача (forward).** Суррогатная модель `f: Synthesis → Properties` предсказывает три
целевые величины по условиям синтеза:

| Таргет | Единицы | Физический смысл |
|---|---|---|
| `E0` | кДж/моль | характеристическая энергия адсорбции (Дубинин–Астахов) |
| `x0` | нм | характеристическая полуширина пор |
| `Sme` | м²/г | удельная поверхность мезопор |

`E0` и `x0` физически связаны соотношением Дубинина–Штёкли `x0 ≈ 12/E0` (нм при кДж/моль), что
делает данные внутренне согласованными (например, `E0=61 ↔ x0≈0.20`; `E0=9.5 ↔ x0≈1.26`).

**Обратная задача (inverse).** Оптимизатор ищет такие условия синтеза, при которых предсказанные
`E0/x0/Sme` максимально близки к целевому профилю, при соблюдении физико-химических ограничений
(порядок температур, температура кипения растворителя, стехиометрия, границы `E0`).

---

## 2. Данные

### 2.1. Происхождение и объём
- Исходник: лабораторные Excel-экспорты изотерм/синтеза → ручная конверсия в
  `data/SEC_SYN_with_features.csv` (54 колонки) → обогащение дескрипторами →
  `data/SEC_SYN_with_features_enriched.csv` (91 колонка).
- Объём: **380 строк** (образцов), пропусков в таргетах нет.
- Провенанс зафиксирован: `data/DATA_DICTIONARY.md` + `data/dataset_manifest.json` (SHA-256 каждого CSV,
  пер-колоночный профиль), генерируются `scripts/build_data_dictionary.py`.
- ⚠️ Конверсия первого Excel → CSV пока внешний ручной шаг (скрипт ингестии не написан — ждёт исходный Excel).

### 2.2. Химическое пространство
- **7 металлов:** Al, Cu, Fe, La, Y, Zn, Zr.
- **4 лиганда:** BDC, BTB, BTC, NH2-BDC.
- **8 растворителей:** ДМФА, ДМСО, Этанол, Ацетонитрил, Вода и смеси (ДМФА/Вода, ДМФА/Этанол, ДМФА/Этанол/Вода).
- **17 групп `Металл|Лиганд`**, сильный дисбаланс:

| Группа | n | Группа | n |
|---|---|---|---|
| Cu\|BTC | 130 | Al\|BDC | 14 |
| Al\|BTC | 83 | Zn\|BTB | 12 |
| Fe\|BDC | 65 | Y\|BTC | 7 |
| Zr\|BDC | 19 | Fe\|NH2-BDC | 5 |
| Fe\|BTC | 18 | Zn\|BDC | 4 |
| La\|BTC | 16 | + 7 групп размера 1–2 | |

→ **73% данных в трёх группах**, 5 синглтонов, 6 групп ≤3 строк.

### 2.3. Особенности данных (критичны для валидации)
- **38 дублей рецептов** (10%): строки с идентичными синтез-входами (часто повторные измерения с
  разными свойствами) → требуют group-aware CV (см. §6.1).
- **Zero-inflation `Sme`:** 31 ноль, 90 значений `<5 м²/г` (24%) — физический спайк «нет мезопор»;
  std=187, max=1450 (тяжёлый хвост) → самый трудный таргет.
- **Диапазоны таргетов:** `E0` 9.5–61.2; `x0` 0.196–1.26; `Sme` 0–1450.
- **Feed-ratio `R_molar`** систематически отличается от формульной стехиометрии каркаса
  (намеренный избыток реагента/модуляция): Cu\|BTC≈1.71 (каркас 1.5), Fe\|BDC≈0.66 (каркас 1.0).

---

## 3. Архитектура сквозного пайплайна

```
data/SEC_SYN_with_features.csv  (сырой CSV из Excel)
  └─ scripts/enrich_descriptors.py            (RDKit + координационная химия)
     → data/SEC_SYN_with_features_enriched.csv
        └─ load_dataset()  →  prepare_forward_dataset()      (признаки X, таргеты y)
           ├─ scripts/train_forward_model.py  --backend catboost   (CatBoost-ансамбль + MAPIE)
           │    → artifacts/forward_models/{*.cbm, metrics.json,
           │       uncertainty_calibrators.joblib, predictions_*.csv}
           ├─ scripts/validate_uncertainty.py            (внутренняя калибровка UQ)
           ├─ scripts/evaluate_forward_holdout.py        (внешний chemistry holdout)
           ├─ scripts/evaluate_chemistry_logo.py         (LOGO-CV + y-scrambling)
           ├─ scripts/feature_stability.py               (диагностика стабильности фич)
           └─ scripts/run_bofire_opt.py                  (обратный дизайн → shortlist)
              → artifacts/predictions_bofire*.csv
```

Ядро-библиотека `src/adsorb_synthesis/`: `constants.py` (источник истины по таргетам/фичам/границам),
`config.py` (гиперпараметры), `data_processing.py` (загрузка+feature engineering),
`data_validation.py` (валидация), `feature_selection.py` (отбор), `forward_modeling.py`
(CatBoost+MAPIE+TabPFN helpers), `holdout_evaluation.py` (chemistry split), `inverse_optimization.py`
(оптимизатор), `physics_losses.py` (физ-штрафы), `molar_masses.py` (молярные массы).

---

## 4. Обработка данных

### 4.1. Загрузка (`load_dataset`)
Последовательность (всё построчно-детерминированно, без статистик датасета → нет утечки):
1. `normalize_synthesis_columns` — нормализация имён колонок (пробелы, варианты знака градуса °/º/ᵒ,
   гомоглифы Latin↔Cyrillic T·C·O…) к каноническим (`SCHEMA_CANONICAL_COLUMNS`).
2. `add_molar_mass_columns` — молярные массы соли/кислоты из справочника `molar_masses.py`.
3. `add_thermodynamic_features` — `Delta_G`, `K_equilibrium` производные (характеристика, **не** вход модели).
4. `_ensure_adsorption_features` — инженерные адсорбционные дескрипторы (`E·Ws`, `E/E0`, `x0·W0`,
   `B_micropore` и т.д.) — это **выходы характеризации**, в признаки forward-модели **не входят**.
5. `add_temperature_categories` — категории `Tsyn/Tdry/Treg`.
6. `add_salt_mass_features` — стехиометрия/концентрации: `R_molar`, `R_mass`, `C_metal`, `C_ligand`,
   `log_C_*`, `Vsyn_m`, `T_range`, `T_activation`, `T_dry_norm`, `Metal_Ligand_Combo`.
7. `add_solvent_polar_descriptors` — дипольный момент, диэлектрическая проницаемость; для **смесей**
   растворителей свойства усредняются по компонентам.
8. `add_physicochemical_descriptors` — «скрытая вода» (`HYDRATION_MAP`), истинные молярности
   (`Molarity_Metal/Ligand/H2O_Hidden`), `Supersaturation_Index`, `Reactor_Loading_g_mL`.
9. `add_interaction_features` — `Metal_Ligand_Size_Ratio`, `Metal_O_Electronegativity_Diff`, `Jahn_Teller_Active`.
10. Валидация (`validate_SEH_data` + `validate_synthesis_data`).

### 4.2. Валидация (`data_validation.py`, режимы `warn`/`strict`, по умолчанию `warn`)
- **Массы/объёмы** соли, кислоты, растворителя `> 0` → иначе **error**.
- **Порядок температур** (`T_dry ≥ T_syn`, `T_reg ≥ T_dry`) → **warning** (не гарантирован).
- **Температура кипения:** `T_syn ≥ bp(растворитель)` → **warning** (сольвотермальный синтез в закрытом
  сосуде штатно превышает атмосферную bp под автогенным давлением); для смесей берётся минимальная bp компонентов.
- **Стехиометрия (feed `R_molar`)** — data-driven политика (аудит #2): глобальные физ-границы
  `[0.05, 10]` → **error**; групповой медианный фактор (`median/4 .. median·4`, для групп ≥6 строк) →
  **warning** (ловит грубые опечатки, не трогает намеренный DOE-разброс). Формульные ratios каркасов
  (`STOICHIOMETRY_REFERENCE`, с источниками: HKUST-1, MOF-5, UiO-66, MIL-53, MIL-100(Fe), MOF-177) — справочник.
- **Согласованность молей** (аудит #11): precomputed `n_соли/n_кислоты` сверяются с `m/MW`; расхождение >2% → **warning**.
- **SEH-соотношения:** `a0 = 28.86·W0` (error при >1%), `E/E0 ≈ 1/3` (warning), `Ws ≥ W0` (error).

Текущее состояние датасета под новой политикой: по `R_molar` **0 error, ~2 warning** (реальный
выброс Al\|BDC=0.132); остальные предупреждения — порядок температур и bp (ожидаемы для сольвотермики).

### 4.3. Признаки прямой модели (`constants.py`)
- **Входы (`FORWARD_MODEL_INPUTS`, 9):** категориальные `Металл`, `Лиганд`, `Растворитель`; непрерывные
  `m(соли)`, `m(кислоты)`, `Vсин.`, `T.син.`, `T.суш.`, `Tрег`.
- **Инженерные (`FORWARD_MODEL_ENGINEERED_FEATURES`, ~20):** стехиометрия, концентрации, температурные,
  физико-химические (см. §4.1 п.6,8).
- **Дескрипторы реагентов (`FORWARD_MODEL_AUGMENTED_FEATURES`):** металл (ионный радиус, электроотрицательность,
  степень окисления, HSAB-жёсткость, электронное сродство), лиганд (число карбоксилов, молекулярная масса,
  ароматические кольца и т.д.), растворитель (полярность), взаимодействия.
- **Исключено (аудит #5):** RDKit 3D/2D-дескрипторы лиганда (`LIGAND_3D/2D_FEATURES`) — при 4 уникальных
  лигандах вырождаются в lookup из 4 строк (модель «запоминает» лиганда → иллюзорная информативность).
  Категориальный `Лиганд` + `carboxyl_groups`/`molecular_weight` несут реальный сигнал.
- **Утечки таргета нет (проверено):** адсорбционные дескрипторы из `E0`/`x0`/`W0`/… (характеризация) в
  forward-набор не попадают; используются только для физ-валидации.

### 4.4. Отбор признаков (`feature_selection.py`)
- Внутри каждого outer-фолда (no-leakage): доменная курация keep/drop → удаление коллинеарных (|r|>0.85) →
  итеративный VIF (>10) → permutation importance → топ-15 + hard-keep curated.
- **По умолчанию (аудит #8): per-fold stability selection** — отбор бутстрэпится по train-части фолда
  (10 ресэмплов), оставляются фичи с частотой ≥0.5. Убирает шум на малых данных (особенно `Sme`).
  Отключение: `--no-stability-selection`.

---

## 5. Модели

### 5.1. Прямая модель — production: CatBoost + MAPIE
- **Алгоритм:** CatBoost-регрессор, отдельный на каждый таргет, с нативной обработкой категориальных фич.
- **Гиперпараметры (`config.py`, тюнинг Optuna 80 trials/таргет):**

| Таргет | iterations | learning_rate | depth | l2_leaf_reg | min_data_in_leaf | subsample | colsample_bylevel |
|---|---|---|---|---|---|---|---|
| `E0` | 1700 | 0.0336 | 8 | 2.20 | 10 | 0.921 | 0.530 |
| `x0` | 1600 | 0.0638 | 7 | 1.846 | 4 | 0.950 | 0.502 |
| `Sme` | 2000 | 0.0469 | 6 | 0.108 | 10 | 0.926 | 0.623 |

  Общее: loss `RMSE`, early stopping 100 раундов (в CV).
- **Production-ансамбль:** 5 членов (Deep Ensemble, разные сиды, шаг 137) на таргет → среднее предсказание.
- **Веса образцов:** `compute_quality_weights` — строки с сильными физ-нарушениями **down**-weight
  (1/(1+penalty), нижняя граница 0.2).
- **UQ — конформные интервалы:** `mapie.CrossConformalRegressor`, метод `plus` (CV+), `alpha=0.10`
  (целевое покрытие 90%), на той же CV-схеме, что и OOF.

### 5.2. Прямая модель — challenger: TabPFN-3
- `tabpfn>=8.0.0`, явный `ModelVersion.V3` (`create_default_for_version`). Трансформер с in-context
  inference (GPU). На внутреннем CV ≈ CatBoost; **нет interval UQ**. Веса v3 требуют принятия лицензии
  Prior Labs + `TABPFN_TOKEN` (ux.priorlabs.ai), не только HF-токена.

### 5.3. Обратная модель
- **Production:** native **BoFire** (`run_bofire_opt.py`) — target-oriented strategy loop с явными
  ограничениями (порядок температур, bp, стехиометрия, границы `E0` = `[5, 65]` кДж/моль). Выход —
  ранжированный diverse-shortlist с `score`, `feasible`, `Pred_*_lo/hi`, `constraint_reasons`.
- **Research:** `run_botorch_mobo.py`, `run_baybe_campaign.py`; `train_inverse_direct.py` — benchmark-baseline;
  `run_bofire_optuna_legacy.py`, `run_bayes_opt.py` — исторические fallback.

---

## 6. Валидация и эксперименты

### 6.1. Внутренняя CV — group-aware (аудит #1)
`StratifiedGroupKFold(5)`, **группы = рецепт** (хэш 9 синтез-входов): дубли рецептов не попадают в
train и valid одновременно (иначе OOF оптимистичен). Стратификация — по металлу + квартилям таргета.
Та же схема передаётся в MAPIE (`PrecomputedSplitCV`). Проверено: 342 уникальных рецепта, **0 пересечений** между фолдами.

### 6.2. Внешний chemistry holdout
`evaluate_forward_holdout.py`: детерминированный split по `Металл|Лиганд` (модель обучается на одних
химиях, проверяется на отложенных), lookup-таблицы строятся только на train (без утечки).

### 6.3. LOGO-CV + y-scrambling (аудит #3)
`evaluate_chemistry_logo.py`: каждая из 17 групп `Металл|Лиганд` по очереди в holdout, обучение на
остальных. Pooled и per-group метрики + **y-scrambling** (перемешивание таргета → нулевой baseline).
Заменяет одиночный шумный hash-split. Прогон на сервере a100 (`--threads 8` — на 380 строках малопоточный
CatBoost быстрее многопоточного).

### 6.4. Прочее
- `validate_uncertainty.py` — внутренняя калибровка UQ (rejection plots, покрытие).
- `feature_stability.py` — диагностика стабильности отбора фич (bootstrap).
- `run_wave2_suite.py` + `benchmark_wave2.py` + `generate_wave2_report.py` — сравнительный контур
  (CatBoost vs TabPFN, BoFire vs BoTorch vs BayBE vs direct).

---

## 7. Результаты (проверено, 2026-06-10)

### 7.1. Внутренний OOF R² (GroupKFold по рецепту, stability selection по умолчанию)

| Таргет | OOF R² | RMSE | До (с утечкой дублей) |
|---|---|---|---|
| `E0` | 0.812 | 4.74 | 0.810 |
| `x0` | 0.812 | 0.097 | 0.817 |
| `Sme` | 0.754 | 92.5 | 0.665 (без stability) / 0.774 (с утечкой) |

Эффект устранения утечки дублей: раздувался именно `Sme`. Эффект stability selection (аудит #8):
`Sme` OOF **0.66 → 0.75** (стабильный набор фич убирает шум), `E0/x0` без изменений.

### 7.2. Неопределённость (CatBoost+MAPIE, цель покрытия 90%)

| Таргет | coverage | gap к цели | норм. ширина (×σ) |
|---|---|---|---|
| `E0` | 99.5% | +9.5% | 1.34 |
| `x0` | 98.7% | +8.7% | 1.35 |
| `Sme` | 98.9% | +8.9% | 1.65 |

→ CV+ переусердствует на малых данных: интервалы шире целевого σ в ~1.3–1.7× (консервативны). KPI
(`interval_coverage_gap`, `interval_width_normalized`) делают это измеримым (аудит #6).

### 7.3. Out-of-chemistry (LOGO-CV, production-итерации, a100)

| Таргет | LOGO R²_pooled (модель) | y-scramble null R² |
|---|---|---|
| `E0` | −0.470 | −0.277 |
| `x0` | −0.743 | −0.107 |
| `Sme` | −0.585 | −0.093 |

→ На незнакомой химии модель **хуже случайного baseline** по всем трём таргетам — не обобщает, а
уверенно экстраполирует в неверную сторону. Строгое подтверждение: **потолок в данных (химическое
разнообразие), а не в алгоритме**.

### 7.4. CatBoost vs TabPFN-3

| Таргет | CatBoost OOF | TabPFN-3 OOF | TabPFN 6.4.1 (старый) | Holdout (оба) |
|---|---|---|---|---|
| `E0` | 0.81 | 0.81 | −3.43 | отрицательный |
| `x0` | 0.81 | 0.80 | −3.46 | отрицательный |
| `Sme` | 0.75 | 0.74 | −0.91 | отрицательный |

→ TabPFN-3 догнал CatBoost на внутреннем CV (резкий скачок против 6.4.1), но out-of-chemistry оба
проваливаются. CatBoost остаётся production (есть MAPIE-интервалы; чуть менее отрицателен на holdout).

---

## 8. Аудит и инженерные решения

Полный аудит пайплайна (12 пунктов) и внедрённые исправления — в
[pipeline_audit_2026-06-09.md](pipeline_audit_2026-06-09.md) и
[pipeline_audit_fixes_2026-06-09.md](pipeline_audit_fixes_2026-06-09.md). Кратко внедрено:
group-aware CV (#1), data-driven валидация стехиометрии (#2), LOGO-CV+y-scrambling (#3), словарь
данных+checksum (#4), убраны вырожденные 3D/2D-дескрипторы (#5), UQ-KPI (#6), границы `E0` (5,65) по
Дубинину (#7), per-fold stability selection как дефолт (#8), ДМСО+смеси растворителей+провенанс (#9),
нормализация имён колонок (#10), проверка согласованности молей (#11), отчёт NaN по фичам (#12 — вскрыл
`T_dry_norm` 58.7% пропусков). Утечки таргета в forward-наборе нет (проверено).

---

## 9. Инфраструктура и воспроизведение

### 9.1. Server-centric workflow (a100)
Тяжёлые прогоны — на сервере; локалка — для правок кода и MCP-навигации. **local ⇄ GitHub ⇄ server на
одном коммите** (правило: код локально → `git push` → `git pull` на сервере; на сервере код не редактируется).
- Путь: `/root/projects/adsorb_synthesis`, клон `git@github.com:DarkPrinceWarrior/adsorption_2025.git`.
- Окружение: `uv venv --python 3.13` + `uv pip install -r requirements.txt`; сервер: uv 0.11.8,
  Python 3.13.5, torch 2.11.0+cu130, CUDA 13.0, 6× A100-40GB (GPU0 занят → `CUDA_VISIBLE_DEVICES=1..5`).
- Запуск в `tmux`, результаты обратно `scp -r a100:.../artifacts/<dir> artifacts/`.

### 9.2. Команды (локально или на сервере с `PYTHONPATH=src`)
```bash
# окружение
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# тесты
PYTHONPATH=src python -m pytest tests/ -v
# 0: обогащение дескрипторами
PYTHONPATH=src python scripts/enrich_descriptors.py --input data/SEC_SYN_with_features.csv --output data/SEC_SYN_with_features_enriched.csv
# 1 (опц.): тюнинг
PYTHONPATH=src python scripts/tune_hyperparams.py --data data/SEC_SYN_with_features_enriched.csv --trials 80
# 2: production forward (CatBoost+MAPIE; stability selection по умолчанию)
PYTHONPATH=src python scripts/train_forward_model.py --data data/SEC_SYN_with_features_enriched.csv --backend catboost
# 3: внутренняя калибровка UQ
PYTHONPATH=src python scripts/validate_uncertainty.py
# 3b: внешний chemistry holdout
PYTHONPATH=src python scripts/evaluate_forward_holdout.py --backend all
# 3c: out-of-chemistry LOGO-CV + y-scrambling
PYTHONPATH=src python scripts/evaluate_chemistry_logo.py --permutations 3
# диагностика стабильности фич
PYTHONPATH=src python scripts/feature_stability.py --bootstraps 30
# провенанс датасета
PYTHONPATH=src python scripts/build_data_dictionary.py
# 4: обратный дизайн (native BoFire)
PYTHONPATH=src python scripts/run_bofire_opt.py --E0 15.0 --x0 0.5 --Sme 100.0 --trials 300 --shortlist-size 12 --output artifacts/predictions_bofire.csv
# 5 (опц.): сравнительный контур wave 2
PYTHONPATH=src python scripts/run_wave2_suite.py --mode full --data data/SEC_SYN_with_features_enriched.csv --catboost-models artifacts/forward_models --E0 15 --x0 0.5 --Sme 100 --run-dir artifacts/wave2_runs/selection_run
```

### 9.3. Структура репозитория
- `src/adsorb_synthesis/` — ядро (см. §3).
- `scripts/` — runnable-точки (enrich, tune, train, validate, holdout, logo, stability, bofire, wave2, …).
- `tests/` — pytest (unit + `test_wave3_regressions.py`); 12/12 проходят.
- `data/` — CSV-датасеты + словарь/манифест.
- `artifacts/` — модели/метрики/прогоны (gitignored): `forward_models` (production CatBoost+MAPIE),
  `forward_models_tabpfn3` (challenger), `forward_logo` (LOGO), `feature_stability`, `forward_holdout*`, `wave2_runs`.
- `docs/` — данный документ, аудит, исправления.

### 9.4. Конвенции
`from __future__ import annotations`; type hints, `X | Y`; `@dataclass`; данные/предсказания — CSV,
метрики/конфиг — JSON, модели/калибраторы — joblib; feature selection строго fold-local.

---

## 10. Ограничения и выводы

1. **Главный потолок — данные, не алгоритм.** Out-of-chemistry генерализация = 0 (LOGO хуже null по всем
   таргетам). Внутренний OOF ~0.75–0.81 относится только к уже виденным химиям.
2. **Малая и несбалансированная выборка:** 380 строк, 17 групп, 73% в трёх, 4 лиганда, 10% дублей рецептов.
3. **UQ-интервалы консервативны** (CV+ на малых данных): покрытие ~99% при цели 90%, ширина ~1.3–1.7σ.
4. **`Sme` зашумлён** (zero-inflation, тяжёлый хвост) — самый трудный таргет даже после stability selection.

**Рекомендация:** рост качества (особенно out-of-chemistry) даст **расширение химического разнообразия
данных** (новые металлы/лиганды/растворители, устранение дублей) — это направление, а не дальнейший
тюнинг модели. Внешний data-effort (ингестия NIST/MOF-датасетов) — отдельная ветка работ.
