# Wave 2 Decision Memo

Дата фиксации: **8 марта 2026**

## Что сравнивали

В runnable `wave2`-контуре были сравнены:

* `CatBoost` vs `TabPFN` для forward stage
* `BoFire` vs `BoTorch` vs `BayBE` для inverse stage
* `direct inverse` как отдельный baseline класса `target -> recipe`

Основание: полный прогон suite в [artifacts/wave2_runs/selection_20260308](/home/ruslan_safaev/adsorb_synth/adsorb_synthesis/artifacts/wave2_runs/selection_20260308).

## Итоговое решение

### Рекомендуемый production stack

* **Forward:** `CatBoost`
* **UQ:** `MAPIE`
* **Inverse:** `BoFire`

### Статус остальных веток

* **TabPFN:** challenger, но не production default
* **BoTorch:** research optimizer
* **BayBE:** campaign-oriented backend
* **Direct inverse:** benchmark-only baseline

## Почему выбран CatBoost

На полном benchmark run `CatBoost` лучше `TabPFN` по всем трем target-ам:

* `E0`: `R2_oof = 0.8100` vs `0.8025`
* `x0`: `R2_oof = 0.8174` vs `0.8054`
* `Sme`: `R2_oof = 0.7739` vs `0.6344`

Дополнительно:

* у `CatBoost` уже встроен production-ready interval UQ через `MAPIE`
* `TabPFN` тяжелее operationally и не даёт преимущества на текущем датасете

## Почему выбран BoFire

На benchmark для target-профиля `E0=15, x0=0.5, Sme=100`:

* `BoFire best_score_pool = 0.1648`
* `BoTorch best_score_pool = 0.1783`
* `BayBE best_score_pool = 0.1842`

У всех трёх backend-ов feasibility была `1.0`, но `BoFire` дал лучший score и остаётся самым практичным production optimizer.

## Почему direct inverse не выбран как основной путь

`direct inverse` показал себя полезным как benchmark, но не как основной workflow:

* feasibility после repair/recheck: `1.000`
* recheck MAE:
  * `E0 = 3.1159`
  * `x0 = 0.0668`
  * `Sme = 64.7440`

То есть baseline способен выдавать допустимые рецепты, но не выигрывает у связки `forward surrogate + optimizer` как механизма подбора рецепта под target values.

## Практический вывод

Если нужна рабочая схема сейчас:

1. обучать production forward pipeline на `CatBoost`
2. валидировать интервалы через `validate_uncertainty.py`
3. подбирать рецепты через `run_bofire_opt.py`

Остальные ветки сохраняются в репозитории как:

* research challengers
* benchmark baselines
* инструменты для последующих сравнений
