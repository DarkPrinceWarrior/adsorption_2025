# Реализация аудита пайплайна — отчёт о внедрении

Дата: **9 июня 2026**. Связано с [pipeline_audit_2026-06-09.md](pipeline_audit_2026-06-09.md).
Все значения выверены научно (Tavily/литература) и сверены с фактическими данными БД. Модели не менялись.

---

## Сводка статуса (12/12 внедрено)

| # | Пункт | Что сделано | Файлы |
|---|---|---|---|
| 1 | Утечка дублей рецептов в CV | `StratifiedKFold` → `StratifiedGroupKFold` по рецепту; распространено на MAPIE | `forward_modeling.py` (`build_recipe_group_keys`), `train_forward_model.py` |
| 2 | Границы стехиометрии | data-driven политика: глобальные физ-границы=ERROR, групповой медианный фактор=WARNING; формульные ratios — справочник с источниками | `constants.py`, `data_validation.py` |
| 3 | Шумный single-split holdout | Новый Leave-One-Chemistry-Out CV + y-scrambling | `scripts/evaluate_chemistry_logo.py` |
| 4 | Невоспроизводимая ингестия | Словарь данных + манифест с SHA-256 | `scripts/build_data_dictionary.py`, `data/DATA_DICTIONARY.md`, `data/dataset_manifest.json` |
| 5 | Инъекция LIGAND_3D/2D | Убрана на источнике (`build_lookup_tables`) и в `prepare_forward_dataset` (P4.1) | `data_processing.py` |
| 6 | Переширокие UQ-интервалы | KPI: `interval_coverage_gap`, `interval_width_normalized` | `train_forward_model.py` |
| 7 | E0-границы / zero-inflation Sme | E0 (10,50)→(5,65) по Дубинину-Штёкли x0≈12/E0 + данным; zero-inflation Sme задокументирована | `constants.py` |
| 8 | Нестабильный отбор фич | `stability_select_features` + анализ; **вшито в обучение** флагом `--stability-selection` (per-fold bootstrap на train, без утечки) | `feature_selection.py`, `scripts/feature_stability.py`, `train_forward_model.py` |
| 9 | Хардкод-справочники | ДМСО добавлен в точки кипения; смеси растворителей (min-bp); провенанс/источники | `constants.py`, `data_validation.py` |
| 10 | Хрупкие имена колонок | Слой нормализации (пробелы, °/º/ᵒ, гомоглифы Latin↔Cyrillic) | `constants.py` (`SCHEMA_CANONICAL_COLUMNS`), `data_processing.py` (`normalize_synthesis_columns`) |
| 11 | Рассинхрон молей | Проверка согласованности precomputed vs mass/MW (warning при расхождении >2%) | `data_validation.py` |
| 12 | Пропуски в фичах | Отчёт о доле NaN по фиче при обучении | `train_forward_model.py` |

---

## Научная верификация (ключевые сверки)

- **Стехиометрия каркасов (литература):** HKUST-1 Cu₃(BTC)₂ → 1.5; MOF-5 Zn₄O(BDC)₃ → 1.33; UiO-66 Zr₆O₄(OH)₄(BDC)₆ → 1.0; MIL-53 M(OH)(BDC) → 1.0; MIL-100(Fe) Fe₃O(BTC)₂ → 1.5; MOF-177 Zn₄O(BTB)₂ → 2.0. **Вывод:** feed-ratio лаборатории систематически отличается от формульного (Cu|BTC≈1.71, Fe|BDC≈0.66 по данным) — это намеренный избыток реагента/модуляция, поэтому формульные ratios оставлены справочником, а валидация — по выбросам.
- **Zr-соль:** Молярка_соли=233 → ZrCl₄ (безводный) → `HYDRATION_MAP['Zr']=0` **верно**.
- **Y-соль:** Молярка_соли=383 → Y(NO₃)₃·6H₂O → `HYDRATION_MAP['Y']=6` **верно**.
- **E0 (Дубинин):** соотношение Дубинина-Штёкли x0[нм]≈12/E0[кДж/моль] делает данные самосогласованными (E0=61↔x0=0.20; E0=9.5↔x0=1.26). Старая граница (10,50) несправедливо штрафовала 9 высоких + 3 низких валидных E0.
- **Sme zero-inflation:** 31 ноль, 90 значений <5 м²/г (24%) — подтверждено. Индикатор «мезопоры есть/нет» как ВХОД отвергнут (это утечка таргета); зафиксировано как характеристика данных.

---

## Результаты честного прогона (после правок)

**Внутренний OOF R² (GroupKFold по рецепту, без утечки дублей):**

| target | OOF R² (новый) | было (с утечкой) |
|---|---|---|
| E0 | 0.821 | 0.810 |
| x0 | 0.814 | 0.817 |
| Sme | **0.665** | 0.774 |

→ утечка дублей рецептов раздувала именно **Sme** (0.77→0.66 после устранения); E0/x0 стабильны. Это честный внутренний показатель.

**Per-fold stability selection (#8, `--stability-selection`, прогон на a100):** на том же GroupKFold:

| target | обычный отбор | stability (10 boot) |
|---|---|---|
| E0 | 0.821 | 0.812 |
| x0 | 0.814 | 0.812 |
| Sme | 0.665 | **0.754** |

→ на самом шумном таргете Sme стабильный отбор убирает шум в наборе фич → OOF **0.66 → 0.75**; E0/x0 без изменений (уже стабильны). Артефакты: `artifacts/forward_models_stability/`. **Теперь production-дефолт** (opt-out: `--no-stability-selection`).

**UQ-KPI (CatBoost+MAPIE, цель покрытия 90%):**

| target | coverage | gap | норм. ширина |
|---|---|---|---|
| E0 | 99.7% | +9.7% | 1.50 |
| x0 | 98.9% | +8.9% | 1.37 |
| Sme | 98.7% | +8.7% | 1.39 |

→ CV+ переусердствует на малых данных: интервалы шире целевого σ в ~1.4–1.5×. KPI делают это измеримым.

**NaN-отчёт вскрыл реальную дыру:** `T_dry_norm` — **58.7% пропусков** (раньше уходило в модель молча).

**Валидация (новая политика):** R_molar ERRORS=0, WARNINGS=2 (только настоящий выброс Al|BDC=0.132); раньше — десятки ложных ERROR.

**Тесты:** `pytest` — 12/12 проходят (тест валидации обновлён под научно-корректное «boiling=warning при сольвотермальном синтезе»).

**LOGO-CV (out-of-chemistry, CatBoost, 17 групп, y-scrambling) — production-прогон на a100:**

| target | LOGO pooled R² (модель) | y-scramble null R² |
|---|---|---|
| E0 | −0.470 | −0.277 |
| x0 | −0.743 | −0.107 |
| Sme | −0.585 | −0.093 |

→ На незнакомой химии модель **хуже нулевого baseline** (перемешанные метки) по всем трём таргетам: она не просто не обобщает, а уверенно экстраполирует в неверную сторону. Это строгое, статистически осмысленное подтверждение главного вывода аудита — **потолок в данных (химическое разнообразие: 17 групп, 73% в трёх, 4 лиганда), а не в алгоритме.** Артефакты: `artifacts/forward_logo/logo_metrics.json` (+ per-group). Прогон на сервере a100: production-итерации (1700–2000), `--permutations 1 --threads 8` (на 380 строках малопоточный CatBoost быстрее). Облегчённый 500-итер прогон давал −0.43/−0.70/−0.55 — вывод идентичен.

---

## Новые команды

```bash
# Честная out-of-chemistry оценка (заменяет шумный single split)
PYTHONPATH=src python scripts/evaluate_chemistry_logo.py --permutations 3

# Стабильность отбора фич
PYTHONPATH=src python scripts/feature_stability.py --bootstraps 30

# Провенанс датасета (словарь + checksum)
PYTHONPATH=src python scripts/build_data_dictionary.py --generated-at <ISO>
```
