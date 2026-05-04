# Ревизия пайплайна inverse design MOF на 4 мая 2026

Дата актуализации: 4 мая 2026 года.

Этот документ описывает текущее состояние кода, данных, постановки задачи, обработки признаков, моделей и результатов. Выводы ниже основаны на текущем коде репозитория, файлах `data/*`, последних артефактах в `artifacts/*` и фактических CLI-точках входа.

## Короткий вывод

Проект сейчас решает задачу target-oriented inverse design для MOF/адсорбентов через двухступенчатую схему:

1. Forward model: рецепт синтеза и химические дескрипторы -> целевые СЭХ.
2. Inverse search: заданные целевые СЭХ -> ранжированный shortlist возможных рецептов.

По состоянию на 4 мая 2026 года основная производственная связка в коде:

- forward backend: CatBoost ensemble;
- неопределенность: MAPIE cross-conformal intervals;
- inverse backend: native BoFire через `scripts/run_bofire_opt.py`;
- challengers/research: TabPFN, BoTorch, BayBE, direct inverse baseline.

Главная инженерная оценка: архитектура задачи поставлена разумно, но качество рекомендаций сейчас ограничено не столько оптимизатором, сколько датасетом. Внутренние OOF/CV-метрики выглядят сильными, но внешний chemistry holdout показывает плохую переносимость на новые химические группы. Поэтому результаты inverse design надо трактовать как ранжирование внутри области похожих исторических рецептов, а не как надежную генерацию новых химий.

## Что предсказываем

Текущий forward target set задается в `src/adsorb_synthesis/constants.py`:

- `E0, кДж/моль` - характеристическая энергия адсорбции;
- `х0, нм` - характеристическая полуширина пор;
- `Sme, м2/г` - удельная поверхность мезопор.

Важно: `W0`, `SБЭТ`, `E`, `Ws` и другие СЭХ больше не являются production-targets forward-модели. Они остаются в датасете и используются для sanity checks/physics penalties, но production forward-модель на 4 мая 2026 предсказывает только `E0`, `х0`, `Sme`.

## Постановка задачи

### Forward problem

Формально:

```text
Recipe + chemistry descriptors -> E0, x0, Sme
```

Базовые входы forward-модели:

- `Металл`
- `Лиганд`
- `Растворитель`
- `m (соли), г`
- `m(кис-ты), г`
- `Vсин. (р-ля), мл`
- `Т.син., °С`
- `Т суш., °С`
- `Tрег, ᵒС`

К ним добавляются производные признаки: стехиометрия, концентрации, молярности, скрытая вода кристаллогидратов, индекс пересыщения, загрузка реактора, физико-химические дескрипторы металла/лиганда/растворителя и interaction-признаки.

### Inverse problem

Пользователь задает целевые значения, например:

```text
E0 = 15
x0 = 0.5
Sme = 100
```

Оптимизатор перебирает/предлагает рецепты, forward-модель предсказывает свойства, затем кандидат получает score:

```text
sum(abs((prediction - target) / max(abs(target), 1)) ** 2)
```

Дополнительно проверяются ограничения:

- порядок/совместимость температур;
- температура синтеза относительно точки кипения растворителя;
- стехиометрия металл/лиганд;
- предсказанный `E0` в допустимых границах.

Итог inverse stage - не один рецепт, а diverse shortlist с `score`, `feasible`, prediction intervals и причинами ограничений.

## Датасет

В репозитории есть два основных CSV:

| Файл | Размер | mtime | Назначение |
|---|---:|---|---|
| `data/SEC_SYN_with_features.csv` | 380 строк, 54 колонки | 2025-10-09 | базовый датасет |
| `data/SEC_SYN_with_features_enriched.csv` | 380 строк, 91 колонка | 2025-11-30 | обогащенный датасет для production |

В обоих файлах все три production-targets заполнены без пропусков.

Распределение по ключевым категориям в enriched dataset:

| Поле | Уникальных | Основная концентрация |
|---|---:|---|
| `Металл` | 7 | `Cu`: 131, `Al`: 100, `Fe`: 89 |
| `Лиганд` | 4 | `BTC`: 255, `BDC`: 102 |
| `Растворитель` | 8 | `ДМФА`: 340 из 380 |
| `Металл` + `Лиганд` | 17 пар | сильно несбалансировано |
| `Металл` + `Лиганд` + `Растворитель` | 29 троек | сильно доминирует ДМФА |

Статистика targets:

| Target | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `E0, кДж/моль` | 24.83 | 10.94 | 9.52 | 19.85 | 61.15 |
| `х0, нм` | 0.576 | 0.223 | 0.196 | 0.605 | 1.261 |
| `Sme, м2/г` | 105.44 | 186.71 | 0.0 | 24.4 | 1450.0 |

Инженерная оценка датасета:

- Данных мало для уверенного extrapolation: всего 380 строк.
- Химическое пространство дискретное и несбалансированное: 7 металлов, 4 лиганда, 8 растворителей, но большая часть строк - ДМФА/BTC/Cu-Al-Fe.
- `Sme` сильно скошен: медиана 24.4, максимум 1450. Это делает RMSE и оптимизацию чувствительными к выбросам.
- `E0` выходит за физический production-bound: в коде диапазон `E0_BOUNDS_KJ_MOL`, а максимум датасета 61.15. Это уже отражается в physics penalty.
- Из-за доминирования повторяющихся химий модель может хорошо интерполировать знакомые группы и плохо переноситься на новые пары/тройки.

## Обогащение данных

Обогащение выполняет `scripts/enrich_descriptors.py`.

Что делает код:

- читает базовый CSV;
- проверяет наличие колонок `Металл` и `Лиганд`;
- строит таблицы ligand descriptors по SMILES;
- считает RDKit 3D descriptors по нескольким конформерам;
- считает 2D chemical descriptors;
- добавляет metal coordination descriptors;
- добавляет interaction features;
- сохраняет enriched CSV.

Важный текущий компромисс: production-модель больше не полагается на полный набор ligand 3D/2D descriptors. В `constants.py` явно зафиксировано, что при 4 уникальных лигандах эти признаки превращаются почти в lookup-таблицу из 4 строк. Поэтому production feature selection оставляет категориальный `Лиганд`, `carboxyl_groups`, `molecular_weight` и ограниченный набор domain-safe признаков.

Это правильное решение для текущего объема данных. Полные RDKit-дескрипторы могут быть полезны снова, если появится существенно больше уникальных лигандов.

## Загрузка и обработка данных

Основная точка загрузки: `load_dataset` в `src/adsorb_synthesis/data_processing.py`.

Фактический порядок обработки:

1. `pd.read_csv`.
2. Добавление молярных масс.
3. Добавление термодинамических helper columns, но только если в датасете уже есть `K_equilibrium`.
4. Проверка/добавление adsorption features.
5. Температурные категории.
6. Salt/mass/concentration features.
7. Solvent polarity descriptors.
8. Physicochemical descriptors.
9. Metal-ligand interaction features.
10. `validate_SEH_data`.
11. `validate_synthesis_data`.

Важно: термодинамика больше не синтезируется из `E0` без измеренного `K_equilibrium`. Код сейчас не придумывает `K_equilibrium` из энергии, что хорошо.

## Признаки

### Базовые recipe-признаки

- массы соли и кислоты;
- объем растворителя;
- температуры синтеза, сушки и регенерации;
- металл, лиганд, растворитель.

### Производные recipe-признаки

Из `add_salt_mass_features` и `add_physicochemical_descriptors`:

- `R_molar`;
- `R_mass`;
- `C_metal`;
- `C_ligand`;
- `log_C_metal`;
- `log_C_ligand`;
- `n_соли`;
- `n_кислоты`;
- `Vsyn_m`;
- `T_range`;
- `T_activation`;
- `T_dry_norm`;
- `Metal_Ligand_Combo`;
- `n_water_hidden`;
- `Ratio_H2O_Metal`;
- `Molarity_Metal`;
- `Molarity_Ligand`;
- `Molarity_H2O_Hidden`;
- `Supersaturation_Index`;
- `Reactor_Loading_g_mL`.

### Domain curation

`feature_selection.py` делит признаки на primary/drop группы. Текущий подход:

- сохранить физически осмысленные признаки: oxidation state, ionic radius, electron affinity, Jahn-Teller flag;
- оставить простые ligand-сигналы: carboxyl groups, molecular weight;
- оставить recipe-сигналы: `R_molar`, `C_metal`, volume;
- оставить температуры;
- убрать дубли и сильные корреляты: `R_mass`, `C_ligand`, `Vsyn_m`, `log_C_ligand`, часть ligand descriptors.

Оценка: эта логика соответствует малому датасету. Для 380 строк лучше иметь меньше устойчивых признаков, чем широкий RDKit/Mordred-пространственный набор, который легко переобучается.

## Валидация данных

На enriched dataset текущий код дает:

- SEH validation: 0 errors, 0 warnings.
- Synthesis validation: 330 errors, 73 warnings.
- Из synthesis errors: 323 связаны с `R_molar`, еще 7 - с превышением точки кипения растворителя.
- Physics report: `e0_bounds_mean = 0.13685`, `energy_ratio_mean = 0`, `ws_w0_mean = 0`.

Это не значит, что пайплайн падает: production scripts используют `validation_mode=warn`. Но это значит, что strict validation сейчас практически неприменим к текущему датасету.

Инженерная оценка:

- Для ML это допустимо как diagnostic layer, если мы честно признаем шум и downweighting.
- Для автоматического лабораторного рецепта это риск: часть historical rows нарушает формализованные правила, а модель учится на них.
- Стехиометрические правила в `STOICHIOMETRY_TARGETS` могут быть слишком жесткими или не совпадать с исторической практикой. 323 нарушения из 380 строк - сигнал, что либо данные реально грязные, либо validation spec не соответствует датасету.

## Train/test и holdout

Для внутренней оценки CatBoost используется 5-fold stratified CV:

- stratification key = металл с rare grouping + target quartile;
- feature selection выполняется fold-local;
- OOF-предсказания сохраняются;
- production-модели потом обучаются на всем датасете.

Для внешней проверки есть chemistry holdout:

- split по `Металл` + `Лиганд`;
- holdout fraction 0.2;
- seed 42;
- последний holdout содержит 83 строки.

Это правильный тип проверки для inverse design, потому что случайный split завышает качество на малых химических таблицах. Именно chemistry holdout показывает реальную проблему переносимости.

## Forward-модели

### Production: CatBoost

Текущий CatBoost-пайплайн:

- per-target tuned hyperparameters из `config.py`;
- 5 OOF folds;
- fold-local curated feature selection;
- sample weights через `compute_quality_weights`: строки с physics penalty не выбрасываются, а downweight-ятся;
- production ensemble из 5 моделей на всем датасете;
- MAPIE `CrossConformalRegressor`, confidence level 90%, method `plus`;
- сохранение `metrics.json`, `feature_meta.joblib`, `uncertainty_calibrators.joblib`, `predictions_*.csv`, `catboost_*.cbm`.

Последние production artifacts:

- `artifacts/forward_models/metrics.json`
- дата: 2026-03-08 17:12:24

Ключевые метрики:

| Target | OOF R2 | OOF RMSE | Production R2 | Production RMSE |
|---|---:|---:|---:|---:|
| `E0` | 0.8100 | 4.7619 | 0.9675 | 1.9686 |
| `х0` | 0.8174 | 0.0952 | 0.9588 | 0.0452 |
| `Sme` | 0.7739 | 88.6581 | 0.9619 | 36.3747 |

Оценка: OOF-метрики хорошие для такого малого датасета. Production-метрики заметно выше OOF, потому что считаются на тех же данных, на которых production ensemble обучен; их нельзя использовать как честную оценку generalization.

### Challenger: TabPFN

TabPFN используется как research/challenger backend:

- OOF 5-fold;
- один production model на target;
- нет prediction intervals;
- признаки предварительно преобразуются в numeric frame.

Оценка: TabPFN в текущем коде полезен для сравнения, но не является production default. Причины: нет UQ layer, слабее CatBoost на holdout, и модель менее интегрирована с inverse uncertainty.

## Uncertainty quantification

CatBoost использует MAPIE cross-conformal intervals:

- internal interval coverage в production artifacts высокая;
- `validate_uncertainty.py` строит rejection plots и coverage plot;
- последний plot: `artifacts/plots/uncertainty_rejection_plots.png`, 2026-03-08 17:13:34.

Но внешний holdout показывает проблему:

| Backend | Target | Holdout R2 | Holdout RMSE | Holdout interval coverage |
|---|---|---:|---:|---:|
| CatBoost | `E0` | -1.8618 | 4.8171 | 0.9759 |
| CatBoost | `х0` | -2.8783 | 0.1758 | 0.4578 |
| CatBoost | `Sme` | -0.9595 | 389.7714 | 0.1687 |
| TabPFN | `E0` | -3.4285 | 5.9923 | n/a |
| TabPFN | `х0` | -3.4550 | 0.1884 | n/a |
| TabPFN | `Sme` | -0.9146 | 385.2761 | n/a |

Оценка: внутренний UQ не калиброван для химического out-of-distribution. Для `E0` interval coverage на holdout высокая, но для `х0` и `Sme` крайне слабая. Это один из главных production-рисков.

## Inverse optimization

### Native BoFire

Production inverse workflow: `scripts/run_bofire_opt.py`.

Код использует:

- `NativeBofireOptimizer`;
- domain из observed categories, observed continuous bounds и observed discrete temperature values;
- `QparegoStrategy` + `qLogNEI`;
- initial experiments из historical/reference rows;
- projection infeasible candidates к nearest feasible reference;
- scoring по forward predictions;
- diverse shortlist selection.

Ограничения проверяются до scoring:

- drying temperature не должна быть слишком выше synthesis temperature;
- drying temperature не должна быть выше regeneration temperature;
- synthesis temperature не должна превышать boiling point solvent;
- `R_molar` должен попадать в bounds;
- predicted `E0` должен быть в допустимом диапазоне.

Оценка: это прагматичная реализация для малого датасета. Особенно важно, что домен строится из наблюдаемых категорий и температур, а не из произвольного химического пространства. Это снижает риск фантазий оптимизатора.

Слабое место: `_project_to_feasible_candidate` заменяет infeasible candidate ближайшим feasible historical reference. Это повышает долю feasible shortlist, но может уменьшать реальную novelty: часть результата становится похожей на существующие рецепты.

### BoTorch

`scripts/run_botorch_mobo.py` - research-grade local BO. Он использует те же targets и forward models, но последний benchmark показывает, что на текущих артефактах BoTorch уступает native BoFire.

### BayBE

`scripts/run_baybe_campaign.py` - campaign-oriented optimizer. Полезен как low-data campaign scaffold, но на последних результатах также уступает BoFire.

### Direct inverse baseline

`scripts/train_inverse_direct.py` учит прямую обратную модель `targets -> recipe`. Это полезный baseline, но не production-подход:

- обратное отображение неоднозначно;
- один target vector может соответствовать множеству рецептов;
- легко получить усредненные или химически сомнительные рецепты;
- оценка все равно проходит через forward-модель, то есть baseline частично замыкается на тот же surrogate.

## Последние результаты inverse

Последний полный suite:

- run dir: `artifacts/wave3_gpu_suite_full`
- manifest: `artifacts/wave3_gpu_suite_full/suite_manifest.json`
- дата: 2026-03-08 20:34:22
- targets: `E0=15`, `x0=0.5`, `Sme=100`

Финальный native benchmark:

- output dir: `artifacts/wave3_benchmark_native_final`
- дата: 2026-03-08 21:19:25

Сводка inverse optimizers:

| Backend | Shortlist rows | Pool rows | Best score shortlist | Mean score shortlist | Feasibility |
|---|---:|---:|---:|---:|---:|
| BoFire native final | 8 | 12 | 0.154140 | 0.256183 | 1.0 |
| BoTorch | 12 | 120 | 0.178292 | 0.221671 | 1.0 |
| BayBE | 8 | 36 | 0.184246 | 0.348547 | 1.0 |

По best score текущий winner - native BoFire.

## Что хорошо сделано

1. Правильная декомпозиция задачи: forward surrogate отдельно от inverse optimizer.
2. Production-target set сужен до `E0`, `х0`, `Sme`, что лучше для малого датасета.
3. Используется fold-local feature selection, что снижает leakage.
4. Ligand 3D/2D descriptors не форсируются в production, что правильно при 4 лигандах.
5. CatBoost подходит для табличного малого датасета с categorical features.
6. Есть conformal intervals и external chemistry holdout.
7. Inverse stage возвращает shortlist с feasibility и diagnostics, а не один "магический" рецепт.
8. Constraints вынесены явно: температура, boiling point, stoichiometry, E0 bounds.

## Что вызывает вопросы

### 1. Synthesis validation конфликтует с историческими данными

330 synthesis errors на 380 строк - это не мелкий шум. Особенно 323 `R_molar` violations. Нужно решить, что верно:

- исторические рецепты действительно нарушают текущие формальные bounds;
- или `STOICHIOMETRY_TARGETS`/tolerances не соответствуют реальным рецептам;
- или расчет `R_molar` через молярные массы/гидраты требует пересмотра.

Пока это не решено, `validation_mode=warn` - вынужденный компромисс.

### 2. Holdout показывает слабую chemistry generalization

Внешний chemistry split дает отрицательные R2 для всех targets. Это означает, что модель в основном интерполирует знакомую химию и плохо переносится на withheld metal-ligand groups.

Это главный bottleneck. Новый оптимизатор не исправит его без новых данных или более надежного descriptor space.

### 3. `Sme` тяжелый target

`Sme` имеет сильную асимметрию и выбросы. RMSE на holdout почти 390, а interval coverage около 17%. Для production decision-making это слабое место.

Возможные улучшения:

- рассмотреть log/robust transform для `Sme`;
- отдельную модель/режим для zero/near-zero `Sme`;
- quantile/conformal calibration отдельно для high-Sme хвоста.

### 4. Feasible projection снижает novelty

Проекция к ближайшему feasible historical reference делает shortlist практически безопаснее, но может возвращать кандидаты, близкие к историческим рецептам. Это приемлемо как conservative production mode, но надо явно называть это "reference-projected optimization".

### 5. UQ пока internal, а не OOD-aware

MAPIE хорошо работает как internal CV diagnostic, но chemistry holdout показывает, что интервалы не надежны для новых химий. Для inverse design стоит добавить OOD/novelty score по chemistry_key/process distance и штрафовать дальние кандидаты.

## Моя оценка выбора модели

CatBoost остается правильным production default на 4 мая 2026 года:

- хорошо работает на малых табличных данных;
- нативно поддерживает категориальные признаки;
- стабилен и интерпретируем;
- совместим с fold-local feature selection;
- production ensemble + MAPIE дают usable uncertainty layer.

TabPFN полезен как challenger, но не как основной backend: нет production UQ, holdout не лучше, и на таком химическом датасете он не решает проблему доменного переноса.

BoFire как основной inverse optimizer также оправдан:

- он лучше последних BoTorch/BayBE результатов по best score;
- хорошо ложится на target-oriented постановку;
- позволяет явно работать с domain/features;
- в коде уже есть constraints и shortlist diversity.

При этом нельзя продавать текущий пайплайн как "модель открывает новые MOF-рецепты". Корректная формулировка: "модель ранжирует feasible candidate recipes в области, близкой к историческим данным, для достижения заданных E0/x0/Sme".

## Рекомендованная формулировка задачи на 4 мая 2026

Текущая корректная постановка:

> По историческому датасету из 380 синтезов MOF/адсорбентов обучается forward surrogate, который по рецепту синтеза и химическим дескрипторам предсказывает `E0`, `х0` и `Sme`. Затем constrained inverse optimizer ищет в области наблюдаемых химий и технологических диапазонов рецепты, минимизирующие normalized target deviation к заданным значениям, с учетом feasibility constraints и uncertainty diagnostics.

Не стоит формулировать как:

> Полностью автономный discovery новых MOF вне области исходных данных.

## Как укрупнить датасет внешними источниками

Поиск по Tavily на 4 мая 2026 года показывает, что внешние данные для MOF есть, но они лежат в разных постановках. Это важно: текущие 380 строк нельзя просто склеить с любой MOF-базой, потому что наш supervised target set - `E0`, `х0`, `Sme`, а большинство публичных источников содержит либо рецепты синтеза без СЭХ, либо структуры/изотермы без рецепта.

Правильная стратегия - собрать не один "большой CSV", а staging-слой из нескольких связанных таблиц:

- `external_synthesis_recipes` - рецепты синтеза и условия;
- `external_adsorption_measurements` - изотермы, газы, температура, давление, loading;
- `external_mof_structures` - CIF, MOFid/MOFkey, CSD refcode, топология, геометрические признаки;
- `mof_identity_crosswalk` - соответствия между MOF name, DOI, CSD refcode, MOFid/MOFkey и локальными `Металл`/`Лиганд`.

После этого можно делать два разных расширения:

1. Расширять пространство рецептов и constraints для inverse search за счет внешних synthesis-протоколов.
2. Расширять target-сигнал только там, где внешние adsorption/property данные можно привести к тем же физическим величинам, что и `E0`, `х0`, `Sme`.

### Источники, которые стоит брать первыми

| Источник | Что внутри | Масштаб | Можно ли укрупнить наш датасет |
|---|---:|---:|---|
| [DigiMOF](https://pmc.ncbi.nlm.nih.gov/articles/PMC10269341/) / [GitHub tools](https://github.com/peymanzmoghadam/DigiMOF-database-master-main) | Текст-майнинг MOF-публикаций: synthesis method, solvent, organic linker, metal precursor, topology | 43 281 статьи, 15 501 уникальный MOF, 52 680 извлеченных synthesis/property связей | Да, это главный источник для расширения рецептурной части. Нужна нормализация металлов, лигандов, растворителей и единиц. Target `E0/x0/Sme` напрямую не дает. |
| [MOF-ChemUnity](https://github.com/AI4ChemS/MOF_ChemUnity) | Knowledge graph CSV: `matching.csv`, `synthesis.csv`, `filtered_experimental_properties.csv`, `computational_properties.csv`, `descriptors.csv`, `water_stability.csv` | Размер зависит от выгрузки, структура уже табличная | Да, очень полезно как crosswalk между DOI/CSD/MOF name и synthesis/properties. Потенциально лучший мост между рецептами, экспериментальными свойствами и структурами. |
| [MOFtextminer](https://github.com/Molsim-Group/MOFtextminer) | Парсер synthesis paragraphs из XML/HTML/PDF, умеет извлекать precursor/linker/solvent/conditions и приводить единицы к SI | Это инструмент, не готовый финальный датасет | Да, как механизм добора новых рецептов из статей и SI. Полезен, если текущие 380 образцов надо системно расширять собственной литературной выборкой. |
| [ESU-MOF](https://arxiv.org/html/2604.20899v1) | Literature-mined synthesis protocols для scale-up prediction: metal precursor, linker, modulators, solvent systems, reaction conditions, scale-up labels | 3 568 synthesis protocols, после dedup 2 684 unlabeled и 723 positive labels | Да, хорошо расширяет synthesis-рецепты и добавляет признак scale-up feasibility. Для `E0/x0/Sme` не является прямой разметкой. |
| [NIST ISODB mirror](https://github.com/NIST-ISODB/isodb-library) | Экспериментальные adsorption isotherms из литературы, GitHub-зеркало с `Library` и `DOI_mapping.csv` | Много материалов и изотерм, не только MOF | Да, но как отдельный adsorption слой. Можно фитить модели изотерм и извлекать сопоставимые параметры, если газ/температура/активация подходят. Рецепта синтеза обычно нет. |
| [MOFX-DB](https://www.nist.gov/publications/mofx-db-online-database-computational-adsorption-data-nanoporous-materials) | Компьютерные adsorption data: H2, CH4, CO2, Xe, Kr, Ar, N2, textural properties, structure files | Более 3 млн simulated adsorption points, свыше 160 000 MOF/zeolite structures | Да для предобучения/дескрипторов/adsorption surrogate, но осторожно: это simulated data, не прямое продолжение экспериментального синтезного датасета. |
| [Northwestern MOF database](https://mof.tech.northwestern.edu/databases) / [API](https://mof.tech.northwestern.edu/api) | CIF + simulated isotherms, bulk downloads по CoREMOF 2014/2019, hMOF, IZA, PCOD-syn, ToBaCCo; textural properties | Крупные bulk-архивы по газам и базам | Да для структурно-адсорбционного слоя. API возвращает MOF objects с `cif`, `isotherms`, gases/elements и bulk zip. Рецептурный слой отсутствует. |
| [CoRE MOF 2024](https://zenodo.org/records/15055758) | Экспериментально опубликованные MOF CIF, CR/NCR классификация, PLD/LCD/PV, surface area, density, topology, OMS, MOFid, stability probabilities | 8 300 SI structures; отдельные CSD-derived наборы крупнее | Да для расширения chemical/structure descriptors и identity matching. Не дает массы, растворители и условия синтеза как в нашем CSV. |
| [QMOF](https://github.com/Andrew-S-Rosen/QMOF) / [Materials Project view](https://contribs.materialsproject.org/projects/qmof) | DFT-свойства MOF: formula, density, PLD/LCD, smiles, MOFid/MOFkey, topology, band gaps, charges, CIF | 20 000+ structures | Не как прямые training rows для текущей задачи. Полезно для transfer learning, descriptors, MOF identity, linker/node representation. |
| [Hugging Face qmof_project](https://huggingface.co/datasets/hermanhugging/qmof_project) | Preprocessed MOF property prediction archives для CGCNN/MGT/PMT: QMOF, ODAC23, hMOF, CoREMOF, MOSAEC | QMOF около 20 372 MOF, ODAC23 около 160k MOF-adsorbate pairs, hMOF около 137k, CoREMOF около 14k | Да для предобучения моделей и feature extraction. Не заменяет текущий табличный supervised датасет рецептов. |
| [MOFTransformer](https://github.com/hspark1212/MOFTransformer) | Transfer learning по porous materials, benchmark properties: H2 uptake, band gap, N2/O2 uptake, CO2 Henry coefficient, thermal stability | От тысяч до десятков тысяч объектов по property task | Полезно как модельная/representation база. Для текущего CatBoost-рецепт -> `E0/x0/Sme` напрямую не подходит без отдельного feature bridge. |

### Что можно реально достать для наших колонок

Прямо сопоставимые с текущими входами признаки:

- `Металл` - из metal precursor, CSD composition, MOF node, formula;
- `Лиганд` - из organic linker, smilesLinkers, linker names;
- `Растворитель` - лучше всего из DigiMOF/MOF-ChemUnity/MOFtextminer/ESU-MOF;
- `Т.син., °С`, время синтеза, method - из synthesis protocols, если парсер извлек условия;
- модификаторы, pH, additives, activation/drying - в текущем датасете частично отсутствуют, но их стоит добавить как новые признаки;
- структурные признаки - PLD, LCD, pore volume, density, topology, OMS, surface area из CoRE/QMOF/MOFDB.

Плохо сопоставимые или требующие ручной нормализации:

- `m (соли), г`, `m(кис-ты), г`, `Vсин. (р-ля), мл` - в литературных рецептах часто есть разные формы записи: mmol, mg, molar ratio, concentration, scale. Нужен парсер единиц и проверка гидратов/солей;
- `Sme, м2/г` - нельзя автоматически заменить на BET surface area из внешней базы. `Sme` в нашем target set - поверхность мезопор, это не то же самое, что total BET/accessible surface area;
- `E0`, `х0` - их можно получить только если есть подходящие adsorption isotherms и мы фитим ту же физическую модель, что использовалась при построении исходного датасета.

### Как укрупнять без порчи постановки

Я бы не делал простой append внешних строк в `data/SEC_SYN_with_features_enriched.csv`. Это сломает смысл target set и создаст ложное ощущение, что у нас стало много supervised examples.

Более надежный план:

1. Сначала собрать `mof_identity_crosswalk`.
   Ключи: normalized MOF name, DOI, CSD refcode, MOFid, MOFkey, formula, metal set, linker smiles/name.

2. Подтянуть рецептурные записи из DigiMOF/MOF-ChemUnity/ESU-MOF.
   Это даст тысячи synthesis protocols и расширит coverage по металлам, лигандам, растворителям, температурам и методам.

3. Подтянуть adsorption/isotherm слой из NIST ISODB, MOFDB/Northwestern, MOFX-DB.
   Для каждой изотермы хранить gas, temperature, pressure grid, loading, source DOI, experimental/simulated flag.

4. Для строк, где есть подходящие изотермы, отдельно фитить параметры, сопоставимые с `E0` и `х0`.
   Только такие строки можно рассматривать как кандидатов для supervised расширения target set. Симуляционные строки лучше помечать `source_type=simulated` и не смешивать с экспериментом без domain weighting.

5. `Sme` расширять отдельно.
   Если внешний источник дает BET/accessible surface area, использовать это как вспомогательный descriptor или auxiliary target, а не как прямую замену `Sme`.

6. Обучать не одну модель, а multi-source схему.
   Для 380 исходных строк оставить high-trust supervised task. Внешние рецепты использовать для feasibility prior, OOD score, candidate generator и representation learning. Внешние isotherms использовать для auxiliary adsorption pretraining и только после физического фитинга - для расширения `E0/x0`.

### Приоритеты интеграции

Самая быстрая польза:

1. MOF-ChemUnity - потому что уже есть CSV-структура и `synthesis.csv`.
2. DigiMOF - потому что это большой источник synthesis properties и растворителей/прекурсоров.
3. CoRE MOF 2024 + QMOF - для MOF identity, MOFid/MOFkey, topology и structural descriptors.
4. NIST ISODB + Northwestern MOFDB - для изотерм и попытки восстановить `E0/x0` через единый fitting pipeline.
5. Hugging Face/QMOF/ODAC/hMOF archives и MOFTransformer - для pretraining/representation learning, а не для прямого добавления строк.

Итоговая оценка: внешний мир позволяет увеличить не 380 -> 500, а собрать десятки тысяч связанных MOF-записей. Но честное supervised расширение именно по нашим трем target-параметрам будет намного меньше и появится только после identity matching и повторного извлечения `E0/x0/Sme` из сопоставимых adsorption/property данных.

## Что я бы делал дальше

1. Разобрать 323 stoichiometry violations:
   - проверить молярные массы солей/кислот;
   - проверить гидратность;
   - сопоставить текущие `STOICHIOMETRY_TARGETS` с реальными literature/practice bounds;
   - разделить "ошибка данных" и "слишком жесткое правило".

2. Сделать отдельный data quality report:
   - по каждой chemistry pair;
   - по растворителям;
   - по target ranges;
   - по validation reasons.

3. Для `Sme` попробовать robust target strategy:
   - log1p target;
   - winsorized diagnostics;
   - отдельный high-Sme regime.

4. Добавить OOD score:
   - distance до ближайшей historical chemistry/process row;
   - penalty в inverse score;
   - отдельную колонку `novelty_risk`.

5. Разделить modes inverse search:
   - conservative/reference-projected;
   - exploratory/native-only;
   - strict-feasible-only.

6. В отчетах явно показывать:
   - internal OOF metrics;
   - external chemistry holdout metrics;
   - feasibility rate;
   - uncertainty coverage на holdout;
   - distance/novelty risk.

7. Запустить отдельный ingestion-проект по внешним MOF-источникам:
   - сначала MOF-ChemUnity/DigiMOF для рецептов;
   - затем CoRE/QMOF/MOFDB для идентификаторов и структурных признаков;
   - затем NIST ISODB/MOFX-DB для изотерм;
   - только после этого расширять supervised target table.

## Итог

На 4 мая 2026 года проект имеет разумную и достаточно зрелую архитектуру для research/decision-support пайплайна: CatBoost forward surrogate, MAPIE uncertainty, constrained BoFire inverse optimization, challenger backends и benchmark/report layer.

Но качество текущих рекомендаций ограничено данными:

- малый объем;
- сильный дисбаланс по химии;
- конфликт validation rules с историческими строками;
- слабая chemistry-holdout переносимость;
- слабая holdout-калибровка uncertainty для `х0` и `Sme`.

Поэтому текущий production вывод должен быть осторожным: использовать shortlist как список кандидатов для экспертной фильтрации и планирования экспериментов, а не как гарантированно оптимальные рецепты.

## Основные кодовые источники

- `src/adsorb_synthesis/constants.py`
- `src/adsorb_synthesis/data_processing.py`
- `src/adsorb_synthesis/data_validation.py`
- `src/adsorb_synthesis/feature_selection.py`
- `src/adsorb_synthesis/forward_modeling.py`
- `src/adsorb_synthesis/holdout_evaluation.py`
- `src/adsorb_synthesis/inverse_optimization.py`
- `src/adsorb_synthesis/physics_losses.py`
- `scripts/enrich_descriptors.py`
- `scripts/train_forward_model.py`
- `scripts/validate_uncertainty.py`
- `scripts/evaluate_forward_holdout.py`
- `scripts/run_bofire_opt.py`
- `scripts/run_botorch_mobo.py`
- `scripts/run_baybe_campaign.py`
- `scripts/train_inverse_direct.py`
- `scripts/benchmark_wave2.py`
- `scripts/generate_wave2_report.py`

## Внешние источники, найденные через Tavily

- DigiMOF: https://pmc.ncbi.nlm.nih.gov/articles/PMC10269341/
- DigiMOF tools: https://github.com/peymanzmoghadam/DigiMOF-database-master-main
- MOF-ChemUnity: https://github.com/AI4ChemS/MOF_ChemUnity
- MOFtextminer: https://github.com/Molsim-Group/MOFtextminer
- ESU-MOF preprint: https://arxiv.org/html/2604.20899v1
- NIST ISODB mirror: https://github.com/NIST-ISODB/isodb-library
- MOFX-DB: https://www.nist.gov/publications/mofx-db-online-database-computational-adsorption-data-nanoporous-materials
- Northwestern MOF database downloads: https://mof.tech.northwestern.edu/databases
- Northwestern MOF database API: https://mof.tech.northwestern.edu/api
- CoRE MOF 2024: https://zenodo.org/records/15055758
- QMOF GitHub: https://github.com/Andrew-S-Rosen/QMOF
- QMOF Materials Project view: https://contribs.materialsproject.org/projects/qmof
- Hugging Face MOF property datasets: https://huggingface.co/datasets/hermanhugging/qmof_project
- MOFTransformer: https://github.com/hspark1212/MOFTransformer
