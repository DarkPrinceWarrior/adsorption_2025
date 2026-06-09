# Wave 2 / Wave 3 Decision Memo

Дата фиксации: **8 марта 2026**

## Что сравнивали

В runnable `wave2`-контуре были сравнены:

* `CatBoost` vs `TabPFN` для forward stage
* `BoFire` vs `BoTorch` vs `BayBE` для inverse stage
* `direct inverse` как отдельный baseline класса `target -> recipe`

Основание:

* исходный runnable suite в [artifacts/wave2_runs/selection_20260308](/home/ruslan_safaev/adsorb_synth/adsorb_synthesis/artifacts/wave2_runs/selection_20260308)
* внешний `chemistry-split` holdout и финальный native inverse benchmark в [artifacts/wave3_benchmark_native_final](/home/ruslan_safaev/adsorb_synth/adsorb_synthesis/artifacts/wave3_benchmark_native_final)

## Итоговое решение

### Рекомендуемый production stack

* **Forward:** `CatBoost`
* **UQ:** `MAPIE`
* **Inverse:** `BoFire` (native strategy loop)

### Статус остальных веток

* **TabPFN:** `TabPFN-3` (`tabpfn>=8.0.0`) — challenger; на внутреннем CV ≈ `CatBoost`, но не production default (нет interval UQ). См. «Обновление Wave 4».
* **BoTorch:** research optimizer
* **BayBE:** campaign-oriented backend
* **Direct inverse:** benchmark-only baseline

## Почему выбран CatBoost

На внутреннем benchmark run `CatBoost` лучше `TabPFN` по всем трём target-ам:

* `E0`: `R2_oof = 0.8100` vs `0.8025`
* `x0`: `R2_oof = 0.8174` vs `0.8054`
* `Sme`: `R2_oof = 0.7739` vs `0.6344`

Дополнительно:

* у `CatBoost` уже встроен production-ready interval UQ через `MAPIE`
* внешний `chemistry-split` holdout используется как основной внешний критерий forward-сравнения
* `TabPFN` тяжелее operationally и не даёт преимущества на текущем датасете

По внешнему holdout обе модели показывают отрицательный `R2`, то есть текущая проблема уже не в orchestration, а в слабой `out-of-chemistry` generalization. Тем не менее:

* `CatBoost` лучше `TabPFN` на `E0` и `x0`
* по `Sme` `TabPFN` чуть лучше, но без production-grade interval UQ

Поэтому production default после wave 3 не меняется: `CatBoost + MAPIE`.

## Обновление Wave 4: TabPFN-3 (9 июня 2026)

После выхода `TabPFN-3` пакет `tabpfn` поднят до `>=8.0.0` (проверено на `8.0.7`); forward-бэкенд явно пинит `ModelVersion.V3` через `create_default_for_version`. Доступ к весам v3 требует разового принятия лицензии Prior Labs (`tabpfn-3-license-v1.0`) и ключа `TABPFN_TOKEN` с https://ux.priorlabs.ai — одного HF-токена недостаточно.

Повторный forward-benchmark (тот же `chemistry-split`, holdout = 83 строки):

| target | CatBoost `R2_oof` | TabPFN-3 `R2_oof` | CatBoost `R2_holdout` | TabPFN-3 `R2_holdout` |
|--------|-------------------|-------------------|-----------------------|-----------------------|
| `E0`   | 0.8100            | 0.8115            | -1.8618               | -2.9300               |
| `x0`   | 0.8174            | 0.8041            | -2.8783               | -3.6091               |
| `Sme`  | 0.7739            | 0.7370            | -0.9595               | -0.9293               |

Выводы:

* `TabPFN-3` совершил качественный скачок на внутреннем CV — с отрицательных `R2_oof` версии `6.4.1` (`E0 -3.43`, `x0 -3.46`, `Sme -0.91`) до паритета с `CatBoost`.
* На внешнем `chemistry-split` holdout оба бэкенда по-прежнему дают отрицательный `R2`: смена forward-модели не решает `out-of-chemistry` generalization (вывод Wave 3 в силе).
* `CatBoost` остаётся production default: чуть менее отрицателен по `E0`/`x0` на holdout и единственный с production-grade interval UQ (`MAPIE`); у `TabPFN`-бэкенда интервалов нет.
* `TabPFN-3` повышается из явного аутсайдера до полноценного challenger.

Артефакты прогона: `artifacts/forward_models_tabpfn3/` (full-fit) и `artifacts/forward_holdout_v3/` (chemistry-split).

## Почему выбран BoFire

На benchmark для target-профиля `E0=15, x0=0.5, Sme=100`:

* `BoFire best_score_pool = 0.1541`
* `BoTorch best_score_pool = 0.1783`
* `BayBE best_score_pool = 0.1842`

У всех трёх backend-ов feasibility была `1.0`, но `BoFire` дал лучший score и наиболее широкий shortlist по chemistry coverage (`5` chemistry groups). После wave 3 он остаётся canonical native production optimizer. Historical wrapper `BoFire domain + Optuna` сохранён отдельно как `run_bofire_optuna_legacy.py`.

## Почему direct inverse не выбран как основной путь

`direct inverse` показал себя полезным как benchmark, но не как основной workflow:

* feasibility после repair/recheck: `1.000`
* recheck MAE:
  * `E0 = 7.6023`
  * `x0 = 0.1733`
  * `Sme = 126.0685`

То есть baseline способен выдавать допустимые рецепты, но не выигрывает у связки `forward surrogate + optimizer` как механизма подбора рецепта под target values.

## Практический вывод

Если нужна рабочая схема сейчас, после всех трёх волн:

1. обучать production forward pipeline на `CatBoost`
2. валидировать internal CV calibration через `validate_uncertainty.py`
3. проверять внешний split через `evaluate_forward_holdout.py`
4. подбирать рецепты через `run_bofire_opt.py`

Остальные ветки сохраняются в репозитории как:

* research challengers
* benchmark baselines
* инструменты для последующих сравнений
