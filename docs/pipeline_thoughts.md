# 🔍 ГЛУБОКИЙ АНАЛИЗ ТЕКУЩЕЙ РЕАЛИЗАЦИИ

Проанализирую систему компонент за компонентом, выявляя сильные стороны и проблемы.

---

## 📊 ОБЩАЯ ОЦЕНКА: 7/10

**Архитектура правильная**, но есть **критические пробелы** между теорией (docs) и реализацией.

---

## 1. АРХИТЕКТУРА (pipeline.py) - 7.5/10

### ✅ Сильные стороны:
1. **Staged pipeline с зависимостями** - правильный подход для sequential prediction
2. **Сохранение/загрузка через joblib** - воспроизводимость
3. **Physics-informed loss через sample weighting** - умно!
4. **Post-processing**: thermodynamics, stoichiometry, temperature order
5. **LookupTables для descriptor imputation**

### ❌ Критические проблемы:

#### 🔴 1.1 МОЛЯРНЫЕ МАССЫ - ОТКУДА?
```python
# В _update_stoichiometry_features:
if {'m (соли), г', 'Молярка_соли'}.issubset(df.columns):
    df['n_соли'] = df['m (соли), г'] / df['Молярка_соли']
```

**КРИТИЧЕСКИЙ БАГ**: 
- Код ожидает колонки `'Молярка_соли'` и `'Молярка_кислоты'`
- **НО НИГДЕ ИХ НЕ СОЗДАЕТ!**
- `build_lookup_tables` пытается их взять из df, но откуда они там?

**Решение**:
```python
# Нужен новый модуль molar_masses.py:
METAL_SALTS_MOLAR_MASSES = {
    'Cu(NO3)2·3H2O': 241.60,
    'Al(NO3)3·9H2O': 375.13,
    ...
}

TYPICAL_SALTS = {'Cu': 'Cu(NO3)2·3H2O', ...}

def add_molar_mass_columns(df: pd.DataFrame):
    df['Молярка_соли'] = df['Металл'].map(
        lambda m: METAL_SALTS_MOLAR_MASSES[TYPICAL_SALTS[m]]
    )
    df['Молярка_кислоты'] = df['Лиганд'].map(LIGAND_MOLAR_MASSES)
```

**И вызывать в load_dataset!**

#### 🔴 1.2 Log-transform - хардкод
```python
# В predict():
if stage.target == "log_salt_mass":  # <-- хардкод!
    results["m (соли), г"] = np.expm1(predictions)
```

**Проблема**: Что если добавим другие log-transformed targets?

**Решение**: Расширить StageConfig:
```python
@dataclass(frozen=True)
class StageConfig:
    ...
    transform: Optional[str] = None  # 'log', 'sqrt', None
    inverse_target: Optional[str] = None  # для обратного преобразования
```

#### 🔴 1.3 N_RATIO_BOUNDS = (0.45, 2.3) - единый для ВСЕХ
```python
def _project_stoichiometry(df: pd.DataFrame):
    lower, upper = N_RATIO_BOUNDS  # (0.45, 2.3)
    target_ratio = ratio.clip(lower, upper)  # <-- НЕПРАВИЛЬНО!
```

**КРИТИЧЕСКАЯ ОШИБКА**: Теряем химическую специфичность!
- HKUST-1 (Cu-BTC): target = **1.5**, tolerance ±10%
- MOF-5 (Zn-BDC): target = **2.0**, tolerance ±10%
- Единый клип (0.45, 2.3) = "все подряд"

**Решение**:
```python
def _project_stoichiometry(df: pd.DataFrame):
    for idx, row in df.iterrows():
        metal, ligand = row['Металл'], row['Лиганд']
        target, tolerance = get_target_stoichiometry(metal, ligand)
        lower, upper = target * (1 - tolerance), target * (1 + tolerance)
        # Проецировать к target, а не просто клипать
```

#### ⚠️ 1.4 Растворитель удален, но features остались
```python
# Комментарий в коде:
# Solvent stage removed: dataset filtered to DMFA only

# НО:
SOLVENT_DESCRIPTOR_FEATURES = [
    'Solvent_MolWt', 'Solvent_LogP', ...  # <-- остались!
]

# И в _ensure_process_defaults:
df['Растворитель'] = 'ДМФА'  # Константа
```

**Проблема**: 
- Если растворитель всегда ДМФА → SOLVENT_DESCRIPTOR_FEATURES тоже константы
- Константы не несут информации → модель их игнорирует
- Зачем они в feature lists?

**Решение**: Удалить из feature_columns ИЛИ явно документировать.

#### ⚠️ 1.5 Temperature - только категории
```python
# Модель предсказывает:
"Tsyn_Category": "Средняя (115-135°C)"

# Но пользователь хочет:
"Т.син., °С": 125  # Число!
```

**Проблема**: Нет обратного преобразования категорий в числа.

**Решение**: Добавить regression stages для numerical temperatures:
```python
StageConfig(
    name="tsyn_numeric",
    target="Т.син., °С",
    problem_type="regression",
    depends_on=(..., "Tsyn_Category"),
)
```

#### ⚠️ 1.6 Physics loss только для classification
```python
if stage.problem_type == "classification" and stage.physics_weight > 0.0:
    estimator_factory = partial(_default_classifier, enable_physics=True)
```

**Вопрос**: А почему не для regression?
- Salt_mass, acid_mass могут быть отрицательными → physics penalty!
- Можно добавить BoundConstraint для масс, объемов

---

## 2. DATA PROCESSING (data_processing.py) - 6/10

### ✅ Сильные стороны:
1. **add_thermodynamic_features** - корректный расчет ΔG и K_eq
2. **_ensure_adsorption_features** - создание производных
3. **Temperature categories**

### ❌ Критические проблемы:

#### 🔴 2.1 add_salt_mass_features - НЕПОЛНАЯ
```python
def add_salt_mass_features(df: pd.DataFrame) -> None:
    df['Metal_Ligand_Combo'] = df['Металл'] + '_' + df['Лиганд']
    df['Log_Metal_MW'] = np.log1p(df['Total molecular weight (metal)'])
    df['Is_Cu'] = (df['Металл'] == 'Cu').astype(int)
    df['log_salt_mass'] = np.log1p(df['m (соли), г'])
```

**ОТСУТСТВУЮТ** (из docs/CHEATSHEET_formulas.txt:157-176):
```python
# Концентрации
C_metal = m(соли) / V_syn
C_ligand = m(кислоты) / V_syn
log_C_metal = log(C_metal)
log_C_ligand = log(C_ligand)

# Мольное соотношение (С ПРАВИЛЬНЫМИ МОЛЯРНЫМИ МАССАМИ!)
R_molar = (m_metal / M_nitrate) / (m_ligand / M_ligand)

# Температурные характеристики
T_range = T_reg - T_syn
T_activation = T_reg - 100

# Проверочные фичи
a0_calc = 28.86 * W0
E_calc = E0 / 3
Ws_W0_ratio = Ws / W0
```

#### 🔴 2.2 build_lookup_tables - предполагает существование данных
```python
metal_table = df[["Металл", *METAL_DESCRIPTOR_FEATURES]].drop_duplicates()
# METAL_DESCRIPTOR_FEATURES содержит 'Молярка_соли'
# НО если этой колонки нет в df → KeyError!
```

**Проблема**: Курица и яйцо.
- Lookup tables создаются из df
- Но в df должны быть дескрипторы
- Дескрипторы должны создаваться ИЗ lookup tables?

**Решение**: Разделить на две функции:
```python
def create_descriptors(df: pd.DataFrame):
    # Создать базовые дескрипторы (молярные массы, и т.д.)
    add_molar_mass_columns(df)
    ...

def build_lookup_tables(df: pd.DataFrame):
    # После создания дескрипторов
    ...
```

#### ⚠️ 2.3 K_equilibrium и Delta_G как ADSORPTION_FEATURES
```python
# В _ensure_adsorption_features:
if 'K_equilibrium' not in df.columns:
    df['K_equilibrium'] = np.exp(
        df['E, кДж/моль'] / (R_kj * TEMPERATURE_DEFAULT_K)  # 298.15K
    )
```

**Проблема**: 
- K_eq зависит от температуры синтеза!
- Использование фиксированных 298.15K - некорректно
- Реальная температура синтеза: 100-150°C

**Решение**: Удалить K_equilibrium из базовых features, вычислять позже с реальной T.

---

## 3. MODERN MODELS (modern_models.py) - 8/10

### ✅ Сильные стороны:
1. **Ensemble из 3 моделей** (TabNet, CatBoost, XGBoost)
2. **Оптимизация весов ансамбля** через SLSQP + L2 regularization
3. **SMOTE/ADASYN** для балансировки
4. **Focal weights** для rare classes
5. **Calibration** (isotonic) для вероятностей
6. **Quantile regression** для robustness
7. **Physics loss integration**

### ❌ Проблемы:

#### 🔴 3.1 Physics loss - O(n) вызовов
```python
def _compute_physics_sample_weights(self, X, base_weights):
    for i in range(X.shape[0]):  # ДЛЯ КАЖДОГО СЭМПЛА!
        sample = X[i:i+1, :]
        loss = self.physics_loss_fn(sample, self.feature_names)
        physics_violations.append(loss)
```

**Проблема**: 
- `combined_physics_loss` возвращает **скаляр** (среднее)
- Вызывается n раз → O(n) создание DataFrame
- Медленно для больших датасетов

**Решение**:
```python
# Переписать physics_loss_fn, чтобы возвращала ВЕКТОР:
def physics_violation_per_sample(X, feature_names, evaluator):
    df = pd.DataFrame(X, columns=feature_names)
    return evaluator.penalties(df)  # УЖЕ вектор!
```

#### 🔴 3.2 Physics loss на transformed данных
```python
# В pipeline._train_and_evaluate:
penalties = physics_violation_scores(physics_frame, evaluator)  # RAW данных

# НО модель получит:
X_transformed = ColumnTransformer.transform(X)  # OneHot + Imputed
```

**КРИТИЧЕСКАЯ ПРОБЛЕМА**:
- Sample weights вычислены на raw features
- Модель обучается на transformed features
- feature_names больше не соответствуют!

**Пример**:
```python
# RAW:
X = ['Металл', 'E0', 'W0']
feature_names = ['Металл', 'E0', 'W0']

# AFTER OneHotEncoder:
X_transformed = ['Металл_Cu', 'Металл_Al', 'E0', 'W0']
# feature_names УСТАРЕЛИ!
```

**Решение**: Вычислять physics penalties ПОСЛЕ preprocessing ИЛИ использовать только numeric columns.

#### 🔴 3.3 TabNet игнорируется при sample_weights
```python
if sw_train is None:  # <-- проверка
    try:
        tabnet = self._make_tabnet()
        tabnet.fit(...)
```

**Проблема**: Если есть sample_weights → TabNet пропускается!

**НО**: PyTorch-TabNet **поддерживает** weights:
```python
tabnet.fit(X_train, y_train, weights=sw_train.flatten())
```

**Решение**: Передавать weights в TabNet.

#### ⚠️ 3.4 _salt_mass_regressor НЕ ИСПОЛЬЗУЕТСЯ
```python
# Определен:
def _salt_mass_regressor(random_state: int):
    return ModernTabularEnsembleRegressor(
        use_quantile=True,
        quantile_alpha=0.5,
    )

# НО в default_stage_configs:
StageConfig(
    name="salt_mass",
    estimator_factory=_default_regressor,  # <-- НЕ _salt_mass_regressor!
)
```

**БАГ**: Специальный регрессор для salt_mass не используется!

#### ⚠️ 3.5 Huber delta = 5.0 для log-space
```python
def _default_regressor(random_state):
    return ModernTabularEnsembleRegressor(
        huber_delta=5.0,  # <-- ДЛЯ log-space
    )
```

**Проблема**: 
- log_salt_mass: range [0.1, 3.0], std = 0.615
- delta = 5.0 → это **8× std**!
- Слишком большой → по сути MAE

**Рекомендация**: delta = 0.6-1.0 для log-space.

---

## 4. PHYSICS LOSSES (physics_losses.py) - 7/10

### ✅ Сильные стороны:
1. **Структурированные constraint классы** (BoundConstraint, ThermodynamicConstraint)
2. **PhysicsConstraintEvaluator** с penalties и summary
3. **project_thermodynamics** корректно реализован
4. **Numpy vectorization** для производительности

### ❌ Проблемы:

#### 🔴 4.1 Только 2 типа constraints
```python
BoundConstraint  # lower ≤ x ≤ upper
ThermodynamicConstraint  # K = exp(-ΔG/RT)
```

**ОТСУТСТВУЮТ** (из docs):
```python
# Точные равенства:
class EqualityConstraint:
    column_a: str
    column_b: str
    coefficient: float  # a = coef × b
    # Пример: a₀ = 28.86 × W₀

# Отношения:
class RelationConstraint:
    column_a: str
    column_b: str
    ratio: float
    tolerance: float
    # Пример: E = E₀ / 3 (±10%)

# Неравенства:
class InequalityConstraint:
    column_a: str
    column_b: str
    type: Literal['>=', '<=']
    # Пример: Ws ≥ W₀
```

#### 🔴 4.2 DEFAULT_PHYSICS_EVALUATOR - неполный
```python
DEFAULT_PHYSICS_EVALUATOR = PhysicsConstraintEvaluator(
    energy_bounds=(
        BoundConstraint("E0, кДж/моль", 10.0, 50.0),
        BoundConstraint("Adsorption_Energy_Ratio", 0.2, 1.0),
    ),
    thermodynamic=ThermodynamicConstraint(...),
)
```

**ОТСУТСТВУЮТ КРИТИЧНЫЕ ПРОВЕРКИ**:
- ✅ a₀ = 28.86 × W₀ (точность 99.9%)
- ✅ E = E₀ / 3 (точность 100%)
- ✅ Ws ≥ W₀ (нарушений 0%)

#### ⚠️ 4.3 Thermodynamic tolerance = 15%
```python
THERMODYNAMIC_TOLERANCE: float = 0.15  # 15%
```

**Вопрос**: Адекватно ли для экспериментальных данных?
- K_eq измеряется с погрешностью
- 15% может быть слишком жестко

**Рекомендация**: Проверить на реальных данных (может быть 20-25%).

---

## 5. CONSTANTS (constants.py) - 6/10

### ❌ Проблемы:

#### 🔴 5.1 K_equilibrium и Delta_G в ADSORPTION_FEATURES
```python
ADSORPTION_FEATURES = [
    'W0, см3/г',
    ...
    'K_equilibrium',  # <-- ПРОИЗВОДНОЕ!
    'Delta_G',        # <-- ПРОИЗВОДНОЕ!
]
```

**Проблема**: Это не базовые СЭХ, а вычисляемые!

Из _ensure_adsorption_features:
```python
df['K_equilibrium'] = np.exp(df['E, кДж/моль'] / (R * 298.15))  # ФИКСИРОВАННАЯ T!
```

**Неправильно**: Реальная T синтеза 100-150°C, не 298.15K!

#### 🔴 5.2 N_RATIO_BOUNDS - единый
```python
N_RATIO_BOUNDS: tuple[float, float] = (0.45, 2.3)
```

**Должен быть**:
```python
STOICHIOMETRY_TARGETS = {
    ('Cu', 'BTC'): {'ratio': 1.5, 'tolerance': 0.10},
    ('Zn', 'BDC'): {'ratio': 2.0, 'tolerance': 0.10},
    ('Al', 'BTC'): {'ratio': 1.0, 'tolerance': 0.10},
    ...
}
```

---

## 6. СКРИПТЫ - 8/10

### ✅ train_inverse_design.py - OK
### ✅ predict_inverse_design.py - OK

### ⚠️ validate_physics_constraints.py - устаревший
```python
parser.add_argument(
    "--data",
    default="data/SEC_SYN_with_features.csv",  # <-- НЕПРАВИЛЬНЫЙ ФАЙЛ!
)
```

Должно быть: `SEC_SYN_with_features_DMFA_only_no_Y.csv`

### ⚠️ tune_inverse_design.py - только metal stage
```python
def evaluate_trial(...):
    metal_metrics = pipeline.stage_results["metal"].metrics
    balanced_accuracy = metal_metrics["balanced_accuracy"]
    return balanced_accuracy, physics_penalty
```

**Проблема**: Оптимизация только по одной стадии!
- Что насчет ligand, salt_mass?

**Решение**: Multi-objective по нескольким стадиям:
```python
metal_acc = metal_metrics["balanced_accuracy"]
ligand_acc = ligand_metrics["balanced_accuracy"]
salt_r2 = salt_metrics["r2"]
combined_score = 0.4*metal_acc + 0.3*ligand_acc + 0.3*salt_r2
```

---

## 7. НЕСООТВЕТСТВИЯ С ТЕОРИЕЙ (docs)

### docs → реализация:

| Формула | Docs | Код | Статус |
|---------|------|-----|--------|
| a₀ = 28.86 × W₀ | ✅ Точность 99.9% | ❌ Не проверяется | **КРИТИЧНО** |
| E = E₀ / 3 | ✅ Точность 100% | ❌ Не проверяется | **КРИТИЧНО** |
| Ws ≥ W₀ | ✅ Нарушений 0% | ❌ Не проверяется | **КРИТИЧНО** |
| R_molar (с M_нитратов) | ✅ Cu(NO₃)₂·3H₂O = 241.60 | ❌ Нет этих данных | **BLOCKING** |
| Стехиометрия специфичная | ✅ HKUST-1 = 1.5 | ❌ Единый clip(0.45, 2.3) | **КРИТИЧНО** |

---

## 8. ПОТЕНЦИАЛЬНЫЕ БАГИ

### 🐛 8.1 МОЛЯРНЫЕ МАССЫ НЕ СОЗДАЮТСЯ
**Blocking bug**: Код требует 'Молярка_соли', но нигде не создает её!

### 🐛 8.2 Physics loss на wrong данных
Sample weights на raw, модель на transformed → feature mismatch!

### 🐛 8.3 _salt_mass_regressor не используется
Определен, но не подключен в stage_configs.

### 🐛 8.4 TabNet игнорируется при weights
Хотя TabNet поддерживает weights через параметр.

### 🐛 8.5 feature_names не передаются вовремя
```python
model = stage.estimator_factory(rng_seed)  # factory НЕ знает feature_names
# Только потом:
if hasattr(model, 'feature_names'):
    model.feature_names = list(stage.feature_columns)
```

Но если в factory создается partial с physics_loss_fn, он уже не получит feature_names!

---

## 📋 ПРИОРИТЕТНЫЙ TODO

### 🔴 BLOCKING (блокирует работу):
1. ✅ **Создать molar_masses.py** с нитратами
2. ✅ **Добавить add_molar_mass_columns** в load_dataset
3. ✅ **Проверить, что датасет загружается**

### 🟠 КРИТИЧНО (сильно влияет на качество):
4. ✅ **Расширить add_salt_mass_features**: C_metal, C_ligand, R_molar, T_range
5. ✅ **Специфичная стехиометрия** для каждого MOF (не единый клип!)
6. ✅ **Использовать _salt_mass_regressor** вместо _default
7. ✅ **Добавить EqualityConstraint**: a₀, E, Ws
8. ✅ **Исправить physics loss**: вектор вместо O(n) вызовов

### 🟡 ВАЖНО (улучшит систему):
9. ⚠️ **Поддержка sample_weights в TabNet**
10. ⚠️ **Numerical temperature regression** stages
11. ⚠️ **Валидация данных** перед обучением
12. ⚠️ **Тесты**: test_stoichiometry, test_validation

---

## 🎯 ИТОГОВЫЙ ВЕРДИКТ

**Код хороший (7/10)**, архитектура правильная, НО:

### Сильные стороны:
- ✅ Правильная staged pipeline с зависимостями
- ✅ Physics-informed через sample weighting
- ✅ Ensemble модели с оптимизацией
- ✅ Post-processing (thermo, stoichiometry)
- ✅ Сохранение/загрузка артефактов

### Критические пробелы:
- ❌ **МОЛЯРНЫЕ МАССЫ НЕ СОЗДАЮТСЯ** (breaking!)
- ❌ Неполный feature engineering
- ❌ Нет проверок точных равенств (a₀, E, Ws)
- ❌ Стехиометрия - единый клип вместо специфичного
- ❌ Physics loss на transformed данных

### Рекомендация:
**Начать с исправления blocking bug (молярные массы)**, затем расширить feature engineering и добавить physics constraints. Это даст максимальный эффект!