import numpy as np
import pandas as pd
from sklearn import metrics

FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}

def feature_label(name):
    return FEATURE_LABELS.get(name, FEATURE_LABELS.get(name.replace(' ', '_'), name))

def validate_features(frame, features):
    if frame.empty:
        raise ValueError('В файле нет строк с винами.')
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.columns.duplicated().any():
        raise ValueError('Названия столбцов не должны повторяться.')
    try:
        result = frame.loc[:, features].apply(pd.to_numeric, errors='raise').astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Все признаки должны быть числами; проверьте формат CSV.') from exc
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Есть пропуски или бесконечные значения. Заполните их перед загрузкой.')
    if 'is_red' in result and not result['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен содержать 0 (белое) или 1 (красное).')
    return result

def predict_batch(model, frame, features, threshold):
    validated = validate_features(frame, features)
    probabilities = model.predict_proba(validated)[:, 1]
    result = validated.copy()
    result['probability_good'] = probabilities
    result['recommended'] = (probabilities >= threshold).astype(int)
    return result

def metric_values(y, probabilities, threshold):
    pred = np.asarray(probabilities) >= threshold
    return {
        'precision': metrics.precision_score(y, pred, zero_division=0),
        'recall': metrics.recall_score(y, pred, zero_division=0),
        'f1': metrics.f1_score(y, pred, zero_division=0),
        'confusion': metrics.confusion_matrix(y, pred, labels=[0, 1]),
    }
