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

* **TabPFN:** challenger, но не production default
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
