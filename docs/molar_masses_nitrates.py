"""
ОБНОВЛЕННЫЕ МОЛЯРНЫЕ МАССЫ ДЛЯ MOF СИНТЕЗА
С учетом использования нитратов-кристаллогидратов
"""

# ==============================================================================
# МОЛЯРНЫЕ МАССЫ СОЛЕЙ МЕТАЛЛОВ (г/моль)
# ==============================================================================

METAL_SALTS_MOLAR_MASSES = {
    # Нитраты меди
    'Cu(NO3)2·3H2O': 241.60,  # Тригидрат нитрата меди(II)
    'Cu(NO3)2·6H2O': 295.65,  # Гексагидрат нитрата меди(II)
    'Cu(NO3)2': 187.56,       # Безводный нитрат меди(II)
    
    # Нитраты цинка
    'Zn(NO3)2·6H2O': 297.49,  # Гексагидрат нитрата цинка
    'Zn(NO3)2': 189.40,       # Безводный нитрат цинка
    
    # Нитраты алюминия
    'Al(NO3)3·9H2O': 375.13,  # Нонагидрат нитрата алюминия
    'Al(NO3)3': 213.00,       # Безводный нитрат алюминия
    
    # Нитраты железа
    'Fe(NO3)3·9H2O': 404.00,  # Нонагидрат нитрата железа(III)
    'Fe(NO3)3·6H2O': 349.95,  # Гексагидрат нитрата железа(III)
    'Fe(NO3)3': 241.86,       # Безводный нитрат железа(III)
    
    # Нитраты циркония
    'ZrO(NO3)2·H2O': 249.26,  # Оксинитрат циркония
    'Zr(NO3)4·5H2O': 429.33,  # Пентагидрат нитрата циркония
    
    # Нитраты церия
    'Ce(NO3)3·6H2O': 434.22,  # Гексагидрат нитрата церия(III)
    'Ce(NO3)3': 326.13,       # Безводный нитрат церия(III)
    
    # Нитраты лантана
    'La(NO3)3·6H2O': 433.01,  # Гексагидрат нитрата лантана
    'La(NO3)3': 324.92,       # Безводный нитрат лантана
    
    # Нитраты иттрия
    'Y(NO3)3·6H2O': 383.01,   # Гексагидрат нитрата иттрия
    'Y(NO3)3': 274.92,        # Безводный нитрат иттрия
}

# Типичные соли, используемые в синтезе MOF (наиболее вероятные)
TYPICAL_SALTS = {
    'Cu': 'Cu(NO3)2·3H2O',    # 241.60 г/моль
    'Zn': 'Zn(NO3)2·6H2O',    # 297.49 г/моль
    'Al': 'Al(NO3)3·9H2O',    # 375.13 г/моль
    'Fe': 'Fe(NO3)3·9H2O',    # 404.00 г/моль
    'Zr': 'ZrO(NO3)2·H2O',    # 249.26 г/моль
    'Ce': 'Ce(NO3)3·6H2O',    # 434.22 г/моль
    'La': 'La(NO3)3·6H2O',    # 433.01 г/моль
    'Y': 'Y(NO3)3·6H2O',      # 383.01 г/моль
}

# ==============================================================================
# МОЛЯРНЫЕ МАССЫ ЛИГАНДОВ (г/моль)
# ==============================================================================

LIGANDS_MOLAR_MASSES = {
    # Карбоксилатные кислоты
    'H3BTC': 210.14,   # Тримезиновая кислота (Benzene-1,3,5-tricarboxylic acid)
    'H2BDC': 166.13,   # Терефталевая кислота (Benzene-1,4-dicarboxylic acid)
    'H3BTB': 446.46,   # 1,3,5-Benzenetrisbenzoic acid
    
    # Альтернативные названия
    'BTC': 210.14,     # То же что H3BTC
    'BDC': 166.13,     # То же что H2BDC
    'BTB': 446.46,     # То же что H3BTB
    
    # Полные названия
    'trimesic_acid': 210.14,
    'terephthalic_acid': 166.13,
}

# ==============================================================================
# ПЕРЕСЧЕТ МОЛЬНЫХ СООТНОШЕНИЙ ДЛЯ НИТРАТОВ
# ==============================================================================

def calc_molar_ratio_nitrates(
    m_metal: float,
    m_ligand: float,
    metal: str = 'Cu',
    ligand: str = 'BTC'
) -> float:
    """
    Расчет мольного соотношения металл:лиганд для нитратов
    
    Параметры:
        m_metal: масса соли металла (г)
        m_ligand: масса лиганда (г)
        metal: символ металла ('Cu', 'Al', 'Fe', и т.д.)
        ligand: тип лиганда ('BTC', 'BDC', 'BTB')
    
    Возвращает:
        R_molar = n(металла) / n(лиганда)
    
    Пример:
        Для Cu-BTC (HKUST-1): теоретически Cu:BTC = 3:2 = 1.5
        m_Cu = 1.0 г Cu(NO3)2·3H2O
        m_BTC = 0.5 г H3BTC
        
        n_Cu = 1.0 / 241.60 = 0.00414 моль
        n_BTC = 0.5 / 210.14 = 0.00238 моль
        R = 0.00414 / 0.00238 = 1.74
    """
    # Определение молярной массы соли
    if metal in TYPICAL_SALTS:
        salt_name = TYPICAL_SALTS[metal]
        M_metal = METAL_SALTS_MOLAR_MASSES[salt_name]
    else:
        raise ValueError(f"Неизвестный металл: {metal}")
    
    # Определение молярной массы лиганда
    if ligand in LIGANDS_MOLAR_MASSES:
        M_ligand = LIGANDS_MOLAR_MASSES[ligand]
    else:
        raise ValueError(f"Неизвестный лиганд: {ligand}")
    
    # Расчет мольных количеств
    n_metal = m_metal / M_metal
    n_ligand = m_ligand / M_ligand
    
    # Мольное соотношение
    return n_metal / n_ligand if n_ligand > 0 else 0.0


# ==============================================================================
# ТЕОРЕТИЧЕСКИЕ СТЕХИОМЕТРИИ MOF
# ==============================================================================

THEORETICAL_STOICHIOMETRY = {
    # Формат: (metal, ligand): (n_metal, n_ligand, название_MOF)
    ('Cu', 'BTC'): (3, 2, 'HKUST-1 / Cu-BTC / MOF-199'),    # Cu3(BTC)2
    ('Cu', 'BDC'): (1, 1, 'MOF-2'),                          # Cu(BDC)
    ('Zn', 'BDC'): (2, 1, 'MOF-5 / IRMOF-1'),               # Zn4O(BDC)3
    ('Al', 'BTC'): (1, 1, 'MIL-100(Al)'),                    # Al3O(BTC)2·nH2O
    ('Fe', 'BTC'): (1, 1, 'MIL-100(Fe)'),                    # Fe3O(BTC)2·nH2O
    ('Zr', 'BDC'): (3, 2, 'UiO-66'),                         # Zr6O4(OH)4(BDC)6
}


def get_theoretical_ratio(metal: str, ligand: str) -> dict:
    """
    Получение теоретического стехиометрического соотношения
    
    Возвращает словарь с:
        - ratio: теоретическое соотношение metal:ligand
        - n_metal: количество атомов металла
        - n_ligand: количество молекул лиганда
        - mof_name: название получающегося MOF
    """
    key = (metal, ligand)
    
    if key in THEORETICAL_STOICHIOMETRY:
        n_metal, n_ligand, mof_name = THEORETICAL_STOICHIOMETRY[key]
        return {
            'ratio': n_metal / n_ligand,
            'n_metal': n_metal,
            'n_ligand': n_ligand,
            'mof_name': mof_name
        }
    else:
        return {
            'ratio': None,
            'n_metal': None,
            'n_ligand': None,
            'mof_name': 'Unknown MOF'
        }


# ==============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==============================================================================

if __name__ == "__main__":
    # Пример расчета для Cu-BTC
    print("=" * 80)
    print("ПРИМЕР: Синтез HKUST-1 (Cu-BTC)")
    print("=" * 80)
    
    # Экспериментальные данные
    m_Cu_salt = 1.485  # г Cu(NO3)2·3H2O
    m_BTC = 0.5025     # г H3BTC
    
    # Расчет мольного соотношения
    R_molar = calc_molar_ratio_nitrates(m_Cu_salt, m_BTC, 'Cu', 'BTC')
    
    print(f"\nИсходные данные:")
    print(f"  Масса Cu(NO3)2·3H2O: {m_Cu_salt} г")
    print(f"  Масса H3BTC: {m_BTC} г")
    
    print(f"\nМолярные массы:")
    print(f"  M[Cu(NO3)2·3H2O] = {METAL_SALTS_MOLAR_MASSES['Cu(NO3)2·3H2O']} г/моль")
    print(f"  M[H3BTC] = {LIGANDS_MOLAR_MASSES['H3BTC']} г/моль")
    
    print(f"\nМольные количества:")
    n_Cu = m_Cu_salt / METAL_SALTS_MOLAR_MASSES['Cu(NO3)2·3H2O']
    n_BTC = m_BTC / LIGANDS_MOLAR_MASSES['H3BTC']
    print(f"  n(Cu) = {n_Cu:.6f} моль")
    print(f"  n(BTC) = {n_BTC:.6f} моль")
    
    print(f"\nЭкспериментальное соотношение:")
    print(f"  Cu:BTC = {R_molar:.3f}")
    
    # Теоретическое соотношение
    theory = get_theoretical_ratio('Cu', 'BTC')
    print(f"\nТеоретическое соотношение для {theory['mof_name']}:")
    print(f"  Cu:BTC = {theory['n_metal']}:{theory['n_ligand']} = {theory['ratio']:.3f}")
    
    print(f"\nИзбыток металла:")
    excess = (R_molar / theory['ratio'] - 1) * 100
    print(f"  {excess:.1f}%")
    
    # Проверка для других металлов
    print("\n" + "=" * 80)
    print("ТЕОРЕТИЧЕСКИЕ СООТНОШЕНИЯ ДЛЯ ДРУГИХ MOF")
    print("=" * 80)
    
    for (metal, ligand), (n_m, n_l, name) in THEORETICAL_STOICHIOMETRY.items():
        ratio = n_m / n_l
        print(f"\n{name}:")
        print(f"  Металл: {metal}, Лиганд: {ligand}")
        print(f"  Соотношение {metal}:{ligand} = {n_m}:{n_l} = {ratio:.3f}")
