from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics
from app_logic import feature_label, predict_batch, metric_values

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')

@st.cache_resource(show_spinner=False)
def load_model(revision):
    # Модель загружается один раз.
    return joblib.load(BASE / 'model.joblib')

@st.cache_data(show_spinner=False)
def load_test(revision):
    return pd.read_csv(BASE / 'test.csv')

@st.cache_data(show_spinner=False)
def test_proba(_model, X, model_revision):
    # _model не хешируется; версия файла входит в ключ кэша.
    return _model.predict_proba(X)[:, 1]

@st.cache_data(show_spinner=False)
def load_config(revision):
    return json.loads((BASE / 'config.json').read_text(encoding='utf-8'))

model_revision = (BASE / 'model.joblib').stat().st_mtime_ns
model = load_model(model_revision)
config = load_config((BASE / 'config.json').stat().st_mtime_ns)
test = load_test((BASE / 'test.csv').stat().st_mtime_ns)
features = config['features']
X_test = test[features]
y_test = test['target'].to_numpy()
probabilities = test_proba(model, X_test, model_revision)

st.title('Винная подборка')
st.caption('Wine Quality. Хорошее вино: оценка эксперта не ниже 7.')
with st.sidebar:
    st.header('Правило отбора')
    threshold = st.slider('Порог рекомендации', 0.0, 1.0, float(config['threshold']),
                          step=0.001, format='%.3f', key='threshold')
    st.caption(f"По умолчанию: {config['threshold']:.3f}. Порог выбран на validation при recall ≥ 0.7.")
    st.info('При высоком пороге в подборку попадает меньше вин. При низком — больше, включая больше посредственных.')

single, model_metrics, batch, explain = st.tabs(['Одно вино', 'Модельные метрики', 'Пакетный прогноз', 'Что выучила модель'])
with single:
    st.subheader('Характеристики вина')
    st.write('Начальные значения — медианы тестовой выборки.')
    values = {}
    columns = st.columns(3)
    for index, name in enumerate(features):
        with columns[index % 3]:
            if name == 'is_red':
                values[name] = int(st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True, key='wine_type') == 'Красное')
            else:
                low, high = float(X_test[name].min()), float(X_test[name].max())
                values[name] = st.number_input(feature_label(name), min_value=low, max_value=high,
                    value=float(X_test[name].median()), step=0.00001 if name=='density' else 0.01, format='%.5f' if name=='density' else '%.3f', key='feature_'+name)
    one = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(one)[0, 1])
    left, right = st.columns(2)
    left.metric('Вероятность хорошего вина', f'{probability:.1%}')
    right.metric('Порог рекомендации', f'{threshold:.3f}')
    if probability >= threshold:
        st.success('Добавить в подборку')
    else:
        st.warning('Оставить вне подборки')

with model_metrics:
    st.subheader('Как порог меняет подборку')
    st.caption(f'{len(y_test):,} вин в отложенной выборке · доля хороших вин {y_test.mean():.1%}')
    values = metric_values(y_test, probabilities, threshold)
    columns = st.columns(4)
    for col, label, value in zip(columns[:3], ['Precision', 'Recall', 'F1'],
                                  [values['precision'], values['recall'], values['f1']]):
        col.metric(label, f'{value:.3f}')
    columns[3].metric('Вин в подборке', str(int(np.sum(probabilities >= threshold))))
    st.caption('Precision: доля хороших вин среди рекомендованных. Recall: доля найденных среди всех хороших. Если рекомендаций нет, precision и F1 равны 0.')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics.ConfusionMatrixDisplay(values['confusion'], display_labels=['Остальное', 'Хорошее']).plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Матрица ошибок')
    metrics.RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1], name='LR')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    tn, fp, fn, tp = values['confusion'].ravel()
    axes[1].scatter(fp/(fp+tn), tp/(tp+fn), color='#a61c3c', s=65, zorder=5, label='Текущий порог')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[2], name='LR')
    axes[2].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайное ранжирование')
    axes[2].scatter(values['recall'], values['precision'], color='#a61c3c', s=65, zorder=5, label='Текущий порог')
    axes[1].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write('Порог меняет решение модели. ROC- и PR-кривые от него не зависят.')
    st.caption('Из-за повторов строк между train и test оценка может быть завышена.')

with batch:
    st.subheader('Прогноз для партии вин')
    st.write('CSV должен содержать 12 признаков без target. Порог — из боковой панели.')
    st.download_button('Скачать пример CSV', (BASE / 'example_wines.csv').read_bytes(), 'example_wines.csv', 'text/csv')
    with st.expander('Обязательные столбцы'):
        st.code(', '.join(features), language=None)
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            result = predict_batch(model, frame, features, threshold)
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as exc:
            st.error(str(exc))
        else:
            st.dataframe(result, hide_index=True, width='stretch')
            st.download_button('Скачать прогноз', result.to_csv(index=False).encode('utf-8-sig'), 'wine_predictions.csv', 'text/csv')
            st.caption('probability_good: вероятность хорошего вина. recommended: решение при текущем пороге (0 или 1).')

with explain:
    st.subheader('Какие признаки влияют на решение')
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh([feature_label(name) for name in coefficients.index], coefficients,
            color=['#a61c3c' if value < 0 else '#24745b' for value in coefficients])
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после StandardScaler'); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    positive = coefficients[coefficients > 0].sort_values(ascending=False).head(3)
    negative = coefficients[coefficients < 0].head(3)
    st.write('Повышают оценку: ' + ', '.join(feature_label(name) for name in positive.index) + '.')
    st.write('Снижают оценку: ' + ', '.join(feature_label(name) for name in negative.index) + '.')
    st.caption('Коэффициент показывает изменение логарифма шансов при росте признака на одно стандартное отклонение, при прочих равных. Причинное влияние из него не следует.')
