"""
ФОРМУЛЫ ДЛЯ ИСПОЛЬЗОВАНИЯ В НЕЙРОННОЙ СЕТИ MOF
Формульно точные связи между параметрами (без подбираемых коэффициентов)
"""

import numpy as np

# ==============================================================================
# I. СВЯЗИ ВНУТРИ ПАРАМЕТРОВ СЭХ (inputs нейросети)
# ==============================================================================

def calc_a0_from_W0(W0: float) -> float:
    """
    Расчет предельной адсорбции из объема микропор
    
    Формула: а0 [ммоль/г] = 28.86 × W0 [см³/г]
    
    Физика: а0 = W0 × (ρ_N2 / M_N2) × 1000
    где ρ_N2 = 0.808 г/см³, M_N2 = 28 г/моль
    
    Точность: MAPE = 0.12% (практически идеальная)
    """
    return 28.86 * W0


def calc_SBET_from_W0_x0(W0: float, x0: float) -> float:
    """
    Расчет удельной поверхности из объема и размера пор
    
    Формула: SБЭТ [м²/г] = 2000 × W0 [см³/г] / x0 [нм]
    
    Физика: Для щелевидных микропор S = 2V/x
    
    Точность: умеренная (зависит от геометрии пор)
    Примечание: работает лучше для материалов с щелевидными порами
    """
    return 2000.0 * W0 / x0


def check_Ws_constraint(Ws: float, W0: float) -> bool:
    """
    Проверка физического ограничения на объемы пор
    
    Формула: Ws [см³/г] ≥ W0 [см³/г]
    
    Физика: Объем супермикропор включает объем микропор
    
    Точность: строгое ограничение, всегда выполняется
    """
    return Ws >= W0


def calc_E0_from_x0(x0: float, k: float = 12.0) -> float:
    """
    Расчет характеристической энергии из размера пор
    
    Формула: E0 ∝ 1/x0
    Упрощенная: E0 ≈ k / x0
    
    Физика: Из уравнения Дубинина-Астахова
    E0 = k × (β/x0)^n, где β - аффинность адсорбата
    
    Параметр k подбирается эмпирически, типично k ≈ 12
    
    Точность: корреляция ≈ 1.0
    """
    return k / x0


def calc_E_from_E0(E0: float) -> float:
    """
    Расчет средней энергии адсорбции из характеристической
    
    Формула: E ≈ E0 / 3
    
    Физика: Эмпирическое соотношение для микропористых материалов
    
    Точность: практически точная (среднее E/E0 = 0.330)
    """
    return E0 / 3.0


# ==============================================================================
# II. СВЯЗИ ВНУТРИ ПАРАМЕТРОВ СИНТЕЗА (outputs нейросети)
# ==============================================================================

def calc_concentration(mass: float, volume: float) -> float:
    """
    Расчет концентрации реагента
    
    Формула: C [г/мл] = m [г] / V [мл]
    
    Точность: 100% (по определению)
    """
    return mass / volume if volume > 0 else 0.0


def calc_mass_ratio(m_metal: float, m_ligand: float) -> float:
    """
    Расчет массового соотношения реагентов
    
    Формула: R_mass = m(соли) / m(кис-ты)
    
    Точность: 100% (по определению)
    """
    return m_metal / m_ligand if m_ligand > 0 else 0.0


def calc_molar_ratio(
    m_metal: float, 
    m_ligand: float,
    M_metal: float = 199.65,  # Cu(acetate)2
    M_ligand: float = 210.14   # H3BTC
) -> float:
    """
    Расчет мольного соотношения реагентов
    
    Формула: R_molar = [m_metal/M_metal] / [m_ligand/M_ligand]
    
    Параметры:
        M_metal: молярная масса соли металла (г/моль)
            - Cu(acetate)2: 199.65 г/моль
            - CuSO4·5H2O: 249.68 г/моль
            - AlCl3: 133.34 г/моль
        M_ligand: молярная масса лиганда (г/моль)
            - H3BTC: 210.14 г/моль
            - H2BDC: 166.13 г/моль
            - H3BTB: 446.46 г/моль
    
    Для HKUST-1 (Cu3(BTC)2): теоретически R_molar = 1.5
    
    Точность: 100% (по определению стехиометрии)
    """
    if m_ligand <= 0 or M_ligand <= 0:
        return 0.0
    
    n_metal = m_metal / M_metal
    n_ligand = m_ligand / M_ligand
    
    return n_metal / n_ligand if n_ligand > 0 else 0.0


# Справочные данные по молярным массам
MOLAR_MASSES = {
    'metals': {
        'Cu(acetate)2': 199.65,
        'CuSO4_5H2O': 249.68,
        'AlCl3': 133.34,
        'FeCl3': 162.20,
        'ZrCl4': 233.04,
    },
    'ligands': {
        'H3BTC': 210.14,  # Benzene-1,3,5-tricarboxylic acid
        'H2BDC': 166.13,  # Benzene-1,4-dicarboxylic acid  
        'H3BTB': 446.46,  # 1,3,5-Benzenetrisbenzoic acid
    },
    'solvents': {
        'DMF': 73.09,
        'Ethanol': 46.07,
        'Water': 18.02,
    }
}

# Температуры кипения растворителей (°C)
BOILING_POINTS = {
    'ДМФА': 153,
    'DMF': 153,
    'Этанол': 78,
    'Ethanol': 78,
    'Вода': 100,
    'Water': 100,
    'Ацетонитрил': 82,
}


def check_temperature_constraint(T_syn: float, solvent: str) -> bool:
    """
    Проверка термодинамического ограничения на температуру синтеза
    
    Условие: T_син < T_кип(растворителя)
    
    Параметры:
        T_syn: температура синтеза (°C)
        solvent: название растворителя
    
    Возвращает True если условие выполнено
    """
    if solvent not in BOILING_POINTS:
        return True  # Неизвестный растворитель, не проверяем
    
    return T_syn < BOILING_POINTS[solvent]


def check_temperature_sequence(T_syn: float, T_dry: float, T_reg: float) -> dict:
    """
    Проверка последовательности температур
    
    Обычная последовательность: T_син ≤ T_суш ≤ T_рег
    
    Возвращает словарь с результатами проверки
    """
    return {
        'T_dry_ge_T_syn': T_dry >= T_syn,
        'T_reg_ge_T_dry': T_reg >= T_dry,
        'sequence_ok': (T_syn <= T_dry <= T_reg),
    }


# ==============================================================================
# III. ВАЛИДАЦИЯ ДАННЫХ
# ==============================================================================

def validate_SEH_data(data: dict) -> dict:
    """
    Валидация параметров СЭХ на физическую корректность
    
    Параметры data должны содержать:
        'W0': объем микропор [см³/г]
        'a0': предельная адсорбция [ммоль/г]
        'Ws': объем супермикропор [см³/г]
        'E0': характеристическая энергия [кДж/моль]
        'E': средняя энергия [кДж/моль]
        'x0': полуширина пор [нм]
        'SBET': удельная поверхность [м²/г]
    
    Возвращает словарь с результатами проверок и ошибками
    """
    errors = []
    warnings = []
    
    # Проверка 1: а0 ≈ 28.86 × W0
    a0_calc = calc_a0_from_W0(data['W0'])
    a0_error = abs(data['a0'] - a0_calc) / data['a0'] * 100
    if a0_error > 1.0:  # > 1% ошибка
        errors.append(f"а0 не соответствует W0: ошибка {a0_error:.2f}%")
    
    # Проверка 2: Ws ≥ W0
    if not check_Ws_constraint(data['Ws'], data['W0']):
        errors.append(f"Нарушено условие Ws >= W0: Ws={data['Ws']}, W0={data['W0']}")
    
    # Проверка 3: E ≈ E0 / 3
    E_ratio = data['E'] / data['E0']
    if abs(E_ratio - 0.33) > 0.10:  # Отклонение > 10% от 0.33
        warnings.append(f"E/E0 = {E_ratio:.3f}, ожидается ~0.33")
    
    # Проверка 4: SБЭТ согласуется с W0/x0
    SBET_calc = calc_SBET_from_W0_x0(data['W0'], data['x0'])
    SBET_error = abs(data['SBET'] - SBET_calc) / data['SBET'] * 100
    if SBET_error > 100:  # > 100% ошибка
        warnings.append(f"SБЭТ сильно отличается от расчетной: {SBET_error:.0f}% (возможно, неидеальная геометрия пор)")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
    }


def validate_synthesis_data(data: dict) -> dict:
    """
    Валидация параметров синтеза на физическую корректность
    
    Параметры data должны содержать:
        'm_metal': масса соли [г]
        'm_ligand': масса лиганда [г]
        'V_syn': объем растворителя [мл]
        'T_syn': температура синтеза [°C]
        'T_dry': температура сушки [°C]
        'T_reg': температура регенерации [°C]
        'solvent': название растворителя
    
    Возвращает словарь с результатами проверок
    """
    errors = []
    warnings = []
    
    # Проверка 1: Положительные массы и объем
    if data['m_metal'] <= 0:
        errors.append("Масса металла должна быть > 0")
    if data['m_ligand'] <= 0:
        errors.append("Масса лиганда должна быть > 0")
    if data['V_syn'] <= 0:
        errors.append("Объем растворителя должен быть > 0")
    
    # Проверка 2: Разумные концентрации
    if data['V_syn'] > 0:
        C_metal = data['m_metal'] / data['V_syn']
        C_ligand = data['m_ligand'] / data['V_syn']
        
        if C_metal > 1.0:  # > 1 г/мл
            warnings.append(f"Очень высокая концентрация металла: {C_metal:.3f} г/мл")
        if C_ligand > 0.5:  # > 0.5 г/мл
            warnings.append(f"Очень высокая концентрация лиганда: {C_ligand:.3f} г/мл")
    
    # Проверка 3: Температурные условия
    if not check_temperature_constraint(data['T_syn'], data['solvent']):
        errors.append(f"T_син ({data['T_syn']}°C) >= T_кип({data['solvent']})")
    
    temp_check = check_temperature_sequence(data['T_syn'], data['T_dry'], data['T_reg'])
    if not temp_check['sequence_ok']:
        warnings.append(f"Нетипичная последовательность температур: T_син={data['T_syn']}, T_суш={data['T_dry']}, T_рег={data['T_reg']}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
    }


# ==============================================================================
# IV. FEATURE ENGINEERING ДЛЯ НЕЙРОСЕТИ
# ==============================================================================

def extract_synthesis_features(data: dict) -> dict:
    """
    Извлечение дополнительных признаков из параметров синтеза
    
    Эти признаки можно добавить к входам нейросети для улучшения обучения
    """
    features = {}
    
    # Концентрации
    features['C_metal'] = calc_concentration(data['m_metal'], data['V_syn'])
    features['C_ligand'] = calc_concentration(data['m_ligand'], data['V_syn'])
    
    # Соотношения
    features['R_mass'] = calc_mass_ratio(data['m_metal'], data['m_ligand'])
    
    # Мольное соотношение (нужны молярные массы)
    if 'M_metal' in data and 'M_ligand' in data:
        features['R_molar'] = calc_molar_ratio(
            data['m_metal'], 
            data['m_ligand'],
            data['M_metal'],
            data['M_ligand']
        )
    
    # Температурные характеристики
    features['T_range'] = data['T_reg'] - data['T_syn']
    features['T_dry_norm'] = (data['T_dry'] - data['T_syn']) / (data['T_reg'] - data['T_syn'] + 1e-6)
    
    return features


def extract_SEH_features(data: dict) -> dict:
    """
    Извлечение дополнительных признаков из параметров СЭХ
    
    Эти признаки можно использовать для:
    1. Проверки консистентности входных данных
    2. Дополнительных входов в нейросеть
    3. Ограничений при обучении (physics-informed)
    """
    features = {}
    
    # Расчетные параметры
    features['a0_calc'] = calc_a0_from_W0(data['W0'])
    features['SBET_calc'] = calc_SBET_from_W0_x0(data['W0'], data['x0'])
    features['E_calc'] = calc_E_from_E0(data['E0'])
    
    # Отношения
    features['Ws_W0_ratio'] = data['Ws'] / data['W0'] if data['W0'] > 0 else 0
    features['E_E0_ratio'] = data['E'] / data['E0'] if data['E0'] > 0 else 0
    
    # Удельные величины
    features['W0_per_SBET'] = data['W0'] / data['SBET'] * 1000 if data['SBET'] > 0 else 0  # см³/м²
    
    return features


# ==============================================================================
# V. PHYSICS-INFORMED LOSS FUNCTIONS
# ==============================================================================

def physics_loss_SEH(y_true, y_pred, alpha: float = 0.1):
    """
    Дополнительная функция потерь с учетом физических ограничений
    
    Параметры:
        y_true: истинные значения [W0, E0, x0, a0, E, SBET, Ws, ...]
        y_pred: предсказанные значения
        alpha: вес физических ограничений
    
    Возвращает комбинированную функцию потерь
    """
    # Индексы параметров (зависят от порядка в выходе сети)
    W0_idx, a0_idx = 0, 3
    E0_idx, E_idx = 1, 4
    Ws_idx = 6
    
    # Базовые потери (MSE)
    base_loss = np.mean((y_true - y_pred) ** 2)
    
    # Физические ограничения
    physics_penalties = 0.0
    
    # 1. а0 должно быть ≈ 28.86 × W0
    a0_calc = 28.86 * y_pred[:, W0_idx]
    physics_penalties += np.mean((y_pred[:, a0_idx] - a0_calc) ** 2)
    
    # 2. E должно быть ≈ E0 / 3
    E_calc = y_pred[:, E0_idx] / 3.0
    physics_penalties += np.mean((y_pred[:, E_idx] - E_calc) ** 2)
    
    # 3. Ws должно быть >= W0 (штраф за нарушение)
    constraint_violation = np.maximum(0, y_pred[:, W0_idx] - y_pred[:, Ws_idx])
    physics_penalties += np.mean(constraint_violation ** 2) * 10  # большой штраф
    
    return base_loss + alpha * physics_penalties


# ==============================================================================
# VI. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==============================================================================

if __name__ == "__main__":
    # Пример данных
    sample_SEH = {
        'W0': 0.5,
        'a0': 14.43,
        'E0': 15.0,
        'E': 5.0,
        'x0': 0.8,
        'SBET': 1250.0,
        'Ws': 0.55,
    }
    
    sample_synthesis = {
        'm_metal': 1.074,
        'm_ligand': 0.491,
        'V_syn': 28.0,
        'T_syn': 130,
        'T_dry': 130,
        'T_reg': 130,
        'solvent': 'ДМФА',
    }
    
    # Валидация
    print("=== Валидация СЭХ ===")
    result_SEH = validate_SEH_data(sample_SEH)
    print(f"Данные корректны: {result_SEH['valid']}")
    if result_SEH['errors']:
        print("Ошибки:", result_SEH['errors'])
    if result_SEH['warnings']:
        print("Предупреждения:", result_SEH['warnings'])
    
    print("\n=== Валидация синтеза ===")
    result_syn = validate_synthesis_data(sample_synthesis)
    print(f"Данные корректны: {result_syn['valid']}")
    if result_syn['errors']:
        print("Ошибки:", result_syn['errors'])
    if result_syn['warnings']:
        print("Предупреждения:", result_syn['warnings'])
    
    # Feature engineering
    print("\n=== Дополнительные признаки из синтеза ===")
    syn_features = extract_synthesis_features(sample_synthesis)
    for key, value in syn_features.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Дополнительные признаки из СЭХ ===")
    SEH_features = extract_SEH_features(sample_SEH)
    for key, value in SEH_features.items():
        print(f"{key}: {value:.4f}")
