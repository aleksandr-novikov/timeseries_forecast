# Прогнозирование и детекция аномалий CPU-утилизации EC2

> Итоговая работа по курсу [TimeSeriesCourse](https://github.com/MVRonkin/TimeSeriesCourse) (автор курса - М. В. Ронкин).
> Датасет: [NAB realAWSCloudwatch](https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch).

## 1. Описание временного ряда

**Источник.** NAB, раздел `realAWSCloudwatch` - метрики CPU реальных EC2-инстансов из CloudWatch.

**Выбранные ряды.** Три CSV из подкатегории `ec2_cpu_utilization_*`:
- `ec2_cpu_utilization_24ae8d.csv`
- `ec2_cpu_utilization_53ea38.csv`
- `ec2_cpu_utilization_5f5533.csv`

**Параметры:**
| Параметр | Значение |
|---|---|
| Целевая переменная (`y`) | CPU utilization, % |
| Частота | 5 минут (`freq='5min'`) |
| Длина каждого ряда | 4032 точки = 14 дней |
| Суточная сезонность | 288 шагов |
| Недельная сезонность | 2016 шагов |
| Пропуски / дубликаты | 0 / 0 |
| Размеченные аномалии (per ряд) | 2 окна (NAB combined_windows.json) |

**Файлы:**
- Сырые данные: `data/raw/NAB/`
- Подготовленный panel (long-format Nixtla): `data/processed/nab_panel.parquet`

## 2. Постановка задачи

**Цель.** Заранее обнаруживать аномалии загрузки CPU (провалы и спайки) и формировать по ним алерт. Горизонт прогноза - сутки (288 шагов), расчет оффлайн, батчами.

**Метрики:**
- **Точечный прогноз:** MAE, RMSE, sMAPE, MASE (сезонность 288)
- **Probabilistic:** scaled_CRPS, coverage@90, scaled_quantile_loss
- **Детекция аномалий:** Precision / Recall / F1 / AUC по разметке NAB

**Бейзлайн:** `SeasonalNaive(288)` (любая более сложная модель должна его побить).

## 3. Запуск

Пошаговый анализ - в ноутбуках `01..05`. Общий код вынесен в `src/`, артефакты сохраняются в `data/processed/`.

```bash
pip install -r requirements.txt
python -m src.pipeline --forecast-model MSTL --anomaly-method zscore_resid
```

**Открыть в Colab:**
[01 EDA](https://colab.research.google.com/github/aleksandr-novikov/timeseries_forecast/blob/main/notebooks/01_EDA_VIS.ipynb) ·
[02 StatForecast](https://colab.research.google.com/github/aleksandr-novikov/timeseries_forecast/blob/main/notebooks/02_StatForecast.ipynb) ·
[03 Diagnostics](https://colab.research.google.com/github/aleksandr-novikov/timeseries_forecast/blob/main/notebooks/03_Diagnostics.ipynb) ·
[04 ML/DL](https://colab.research.google.com/github/aleksandr-novikov/timeseries_forecast/blob/main/notebooks/04_ML_DL.ipynb) ·
[05 Anomaly + Pipeline](https://colab.research.google.com/github/aleksandr-novikov/timeseries_forecast/blob/main/notebooks/05_Anomaly_Pipeline.ipynb)

Первой ячейкой в Colab выполнить `!git clone … && %cd … && !pip install -r requirements.txt`.

## 4. EDA

Графики - в `notebooks/01_EDA_VIS.ipynb` и `report/figures/01_*.png`.

![Три ряда CPU; оранжевым отмечены размеченные окна аномалий NAB](report/figures/01_overview.png)

1. Регулярная сетка, шаг 5 минут, пропусков нет.
2. **Суточная сезонность** во всех трех рядах, амплитуда ~30% CPU. Суточный профиль разный: у одного ряда пик ночью, у остальных днем.
3. **Недельная сезонность** на 14 днях статистически слабая. В `MSTL` ее учитываю, но как надежный сигнал не рассматриваю.
4. **Стационарность** (ADF/KPSS): по рядам результаты разные. Для ARIMA пришлось бы брать `d=1` или `D=1`.
5. **PACF**: 2-4 значимых лага у нуля → ARIMA AR-порядок небольшой.
6. **Связи между рядами** слабые (корреляция STL-остатков ~0.05-0.2). Моделирую ряды по отдельности, но храню в общем panel - для глобальных ML/DL моделей.
7. **Аномалии** раскиданы по всем 14 дням. В test (последние H=288) попадает 100% аномалий одного ряда и ~50% у двух других.

## 5. Анализ аномалий

Детали - в `notebooks/05_Anomaly_Pipeline.ipynb`.

Запускаю 5 unsupervised-детекторов на всем panel и сравниваю по разметке NAB.

| Метод | Параметры | Идея |
|---|---|---|
| Probabilistic interval (MSTL) | level=99 | точка вне 99% PI → аномалия |
| Modified Z-score (residuals) | k=3, median+MAD | robust к выбросам |
| IsolationForest | contamination=0.1, lags 1..288 | random-forest split |
| Local Outlier Factor | n_neighbors=50 | плотностный |
| Matrix Profile (stumpy) | window=288 | поиск редких паттернов |

### Реальные результаты на разметке NAB

| Метод | Precision | Recall | F1 | AUC | n_pred |
|---|---|---|---|---|---|
| **MatrixProfile (победитель)** | **0.194** | **0.251** | **0.219** | 0.568 | 1558 |
| LOF | 0.193 | 0.194 | 0.193 | 0.552 | 1209 |
| IsolationForest | 0.190 | 0.191 | 0.191 | 0.551 | 1209 |
| ZScore_resid (MSTL) | 0.151 | 0.032 | 0.053 | 0.506 | 259 |
| ProbInterval_MSTL@99% | 0.143 | 0.015 | 0.027 | 0.503 | 126 |

**Победитель:** Matrix Profile с F1=0.219.
Абсолютные F1 низкие, но для unsupervised на NAB это норма: окна разметки широкие, точно попасть трудно. У MatrixProfile/LOF/IsolationForest AUC выше 0.55 - все же лучше случайного.
Файл: `data/processed/anomaly_methods_comparison.csv`.

![Matrix Profile: красные крестики - предсказанные аномалии, красные окна - разметка NAB](report/figures/05_anomalies_MatrixProfile.png)

**Дополнительно:** KS-test на остатках MSTL обнаруживает точки concept drift.

![Точки concept drift (KS-test на остатках MSTL)](report/figures/05_drift.png)

## 6. Сводная таблица методов (фактические результаты)

Горизонт прогноза `h = 288` шагов (1 сутки), оценка на последних 288 точках каждого ряда.

![Средний MAE по моделям (3 ряда); пунктир - baseline SeasonalNaive](report/figures/06_mae_comparison.png)

### Stats (mean MAE по 3 рядам, single hold-out)

| Модель | MAE |
|---|---|
| **SNaive (baseline)** | **0.281** |
| HW (HoltWinters) | 0.288 |
| Naive | 0.296 |
| AutoTheta | 0.327 |
| MSTL (daily) | 0.340 |
| RWD | 0.357 |
| MSTL (daily+weekly) | 0.858 |
| AutoETS (ZZA) | 2.070 |

### ML (mean MAE по 3 рядам)

| Модель | MAE |
|---|---|
| **XGB (победитель)** | **0.264** |
| LGBM | 0.269 |
| RF | 0.274 |
| Ridge | 0.421 |
| Lasso | 0.495 |

### DL (mean MAE по 3 рядам, max_steps=50 на MPS)

| Модель | MAE |
|---|---|
| **NHITS (победитель)** | **0.249** |
| LSTM | 0.290 |
| DLinear | 0.292 |

### Победители групп → используются в pipeline

```json
{"stat": "SNaive", "ml": "XGB", "dl": "NHITS", "h": 288}
```

NHITS дал лучший MAE среди всех групп (0.249 против 0.281 у SNaive), но отрыв от бейзлайна небольшой - на 14-дневной истории это ожидаемо.

![Прогноз победителей групп против факта на тестовых сутках; оранжевое окно - test](report/figures/06_forecast_vs_actual.png)

### Зафиксированный набор протестированных моделей

| # | Группа | Модель | Режим | Назначение |
|---|---|---|---|---|
| 0 | Baseline | `SeasonalNaive(288)` | - | минимальный порог качества |
| 1 | Stats | `Naive` | - | контроль |
| 2 | Stats | `RandomWalkWithDrift` | - | контроль |
| 3 | Stats | `AutoETS(model="ZZA")` | авто | экспоненциальное сглаживание |
| 4 | Stats | `AutoTheta(additive)` | авто | классический бенчмарк |
| 5 | Stats | `HoltWinters(error="A")` | ручной | аддитивная сезонность |
| 6 | Stats | `MSTL([288])` | ручной | суточная декомпозиция |
| 7 | Stats | `MSTL([288, 2016])` | ручной | мульти-сезонная декомпозиция |
| 8 | Stats | `AutoARIMA` (1 ряд, non-seasonal) | авто | автоподбор ARIMA |
| 9 | Stats | `Prophet` | ручной | trend + Fourier seasonality |
| 10 | ML | `Ridge` | - | линейная на лагах + кал. фичи |
| 11 | ML | `Lasso` | - | разреженная линейная |
| 12 | ML | `RandomForest` | - | нелинейная без бустинга |
| 13 | ML | `XGBoost` | - | gradient boosting |
| 14 | ML | `LightGBM` | - | gradient boosting |
| 15 | DL | `LSTM` | - | encoder-decoder RNN |
| 16 | DL | `NHITS` | - | нейр. иерархич. интерполяция |
| 17 | DL | `DLinear` | - | декомпозиция через линейные слои |

Сезонную ARIMA (`season_length=288`) не использую - один fit на 5-минутном ряду занимает больше 30 минут. ARIMA представлена только `AutoARIMA` без сезонности на одном ряду, сезонность закрывает `MSTL`. `TBATS` по той же причине в сравнение не вошел, в пайплайне оставлен как опция.

Метрики `MAE / RMSE / sMAPE / MASE / CRPS / coverage@90 / время fit / время predict` - в parquet/CSV в `data/processed/`.

## 7. Pipeline и его тестирование

**Файл:** `src/pipeline.py`. **API/CLI:**

```bash
python -m src.pipeline --forecast-model {SeasonalNaive,MSTL,AutoETS,TBATS} \
                      --anomaly-method {zscore_resid,isolation_forest,lof,matrix_profile} \
                      --horizon 288
```

**Шаги:**
1. Идемпотентно скачивает NAB CSV + JSON меток
2. Собирает panel, делает train/test split (последние H точек = test)
3. Обучает выбранную stat-модель + строит forecast с level=99
4. Считает residuals по in-sample fitted values
5. Запускает выбранный детектор аномалий
6. Сравнивает с ground truth (NAB labels) → P/R/F1/AUC
7. Сохраняет артефакты в `data/processed/`:
   - `pipeline_forecast.parquet`
   - `pipeline_anomalies.parquet`
   - `pipeline_forecast_metrics.csv`
   - `pipeline_summary.json` (включает time fit/forecast/detect)

**Фактический пример запуска** (`data/processed/pipeline_summary.json`, `--forecast-model SeasonalNaive --anomaly-method zscore_resid`):
```json
{
  "forecast_model": "SeasonalNaive", "anomaly_method": "zscore_resid",
  "horizon": 288,
  "fit_time_s": 2.635, "forecast_time_s": 2.334, "detect_time_s": 0.002,
  "forecast_metrics_avg": {"mae": 0.281, "rmse": 0.404, "smape": 0.053, "mase": 0.808},
  "anomaly_scores": {"precision": 0.068, "recall": 0.119, "f1": 0.087, "auc": 0.470}
}
```
(`forecast_metrics_avg` - среднее метрики по трем рядам.)

**Тестирование пайплайна.** Прогнал 5 конфигураций (прогнозная модель × детектор), полный panel, горизонт 288:

| forecast | detector | MAE | Precision | Recall | F1 | fit, s | detect, s |
|---|---|---|---|---|---|---|---|
| **SeasonalNaive** | **matrix_profile** | **0.281** | **0.194** | **0.251** | **0.219** | 2.6 | ~10* |
| SeasonalNaive | zscore_resid | 0.281 | 0.068 | 0.119 | 0.087 | 2.6 | 0.0 |
| MSTL | isolation_forest | 0.858 | 0.234 | 0.118 | 0.157 | 9.6 | 0.7 |
| MSTL | matrix_profile | 0.858 | 0.194 | 0.251 | 0.219 | 9.5 | 0.1 |
| MSTL | zscore_resid | 0.858 | 0.108 | 0.086 | 0.095 | 9.7 | 0.0 |

Лучшая связка - `SeasonalNaive` + `MatrixProfile`: у SeasonalNaive меньший MAE (0.281 против 0.858 у MSTL на этих рядах), а MatrixProfile дает лучший F1 среди детекторов и обучается быстрее (~2.6s против ~9.6s).
\*detect-time у matrix_profile включает numba-JIT при первом вызове (~10s), далее ~0.1s.

Дополнительно:
- **Стресс-тест**: при добавлении искусственной волатильности бейзлайны деградируют слабее гладких моделей.
- **Drift**: KS-test на остатках MSTL отмечает структурные сдвиги (график в §5).
- **Воспроизводимость**: `random_state=42` во всех ML-моделях.

## 8. Выводы

- **Прогноз.** Лучший MAE у NHITS (0.249), затем XGB (0.264); обе модели обходят бейзлайн `SeasonalNaive` (0.281), но разрыв небольшой. На 14-дневной истории это ожидаемо: сложным моделям не хватает данных, и сезонный бейзлайн остается сильным.
- **Что не сработало.** `AutoETS(ZZA)` (MAE 2.07) и `MSTL(daily+weekly)` (0.86) заметно хуже остальных - недельная сезонность на двух неполных циклах скорее мешает.
- **Аномалии.** Лучший детектор - Matrix Profile (F1 0.219, AUC 0.57). Абсолютные значения низкие, но это типично для unsupervised на широких окнах NAB. Детекция через прогнозные интервалы работает слабо (F1 0.03-0.05).
- **Прод.** `SeasonalNaive` обучается ~2.5s и устойчив к шуму - разумный дефолт для батч-прода; NHITS дает небольшой прирост точности ценой обучения. Итоговый выбор пайплайна: прогноз `SeasonalNaive`/`NHITS` + детектор `MatrixProfile`.

## Ссылки

- Курс: <https://github.com/MVRonkin/TimeSeriesCourse>
- Nixtla docs: <https://nixtlaverse.nixtla.io/>
- NAB: <https://github.com/numenta/NAB>
- Hyndman FPPPY: <https://otexts.com/fpppy/>

---

*Числовые результаты и графики - в ноутбуках `notebooks/01..05` и `report/figures/`.*
