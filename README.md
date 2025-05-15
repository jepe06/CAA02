
# CPI Anomaly Detection Across Countries — Deep Learning Pipeline

## 🧠 Goal
Detect anomalous consumer price index (CPI) behavior across multiple countries using deep learning models (LSTM, Transformer, etc.) on time series data — and validate anomalies via real-world news.

https://www.worldbank.org/en/research/brief/inflation-database

---

## ✅ TODO LIST

### 1. 📦 Data Acquisition

#### A. Primary Sources (CPI Data)
- [OECD Stats](https://stats.oecd.org/)
  - Navigate to: **Prices > Consumer Prices (MEI)**
  - Monthly data, clean format.
- [IMF IFS](https://data.imf.org/)
  - Economic Indicators → Prices → CPI, All Items
  - Export via query builder (CSV/Excel).
- [World Bank](https://data.worldbank.org/indicator/FP.CPI.TOTL)
  - Check for **monthly availability** per country.
  - Otherwise fallback to annual.
- [Trading Economics API](https://tradingeconomics.com/)
  - Optional: Real-time data via API (limited free access).
- [UN Data](http://data.un.org/)
  - Backup source, sometimes incomplete.

#### B. Output Format
Target format (wide format matrix):
```
Date        | USA  | Germany | Argentina | ...
------------|------|---------|-----------|-----
2020-01     | 256  | 105     | 345       |
2020-02     | 257  | 106     | 360       |
...
```

---

### 2. 🔧 Preprocessing
- Interpolate or drop missing values.
- Apply per-country normalization:
  - z-score (mean 0, std 1), or
  - min-max (0–1 range).
- Optional: differencing (`ΔCPI`) to remove trends.

---

### 3. 🏗️ Model Architectures

#### A. Model Option 1: LSTM Autoencoder
- Input: `[batch, time_window, num_countries]`
- Encoder → bottleneck → decoder
- Loss: MSE between input and reconstructed window
- High reconstruction error = anomaly

#### B. Model Option 2: Variational Autoencoder (VAE)
- Adds probabilistic representation.
- Use latent mean/variance for soft anomaly scores.

#### C. Model Option 3: Transformer (Informer, TimesNet, etc.)
- Better for long sequences
- Requires positional encoding
- Higher compute cost

#### D. Model Option 4: 1D-CNN / TCN
- Capture short-term patterns efficiently
- Less memory usage vs RNNs

---

### 4. 🧪 Anomaly Detection Logic
- Slide window over full time series (e.g., 12-month lookback)
- For each window:
  - Compute reconstruction error (e.g., MSE)
  - Normalize errors per country
  - Threshold: 
    - `score > mean + 2σ` or use percentile (e.g., 95th)
- Output:
```json
[
  {"country": "Argentina", "date": "2019-08", "score": 3.92},
  {"country": "Turkey", "date": "2022-01", "score": 4.11},
  ...
]
```

---

### 5. 🧠 Model Evaluation (No Labels)

#### A. Human-in-the-Loop via ChatGPT
- After anomaly list is created, send to ChatGPT.
- For each `(country, date)`:
  - News lookup and real-world event matching.
  - ChatGPT returns a label: "likely real anomaly" or "false positive".

#### B. Manual Validation Metrics
- Top-K anomalies → count real ones.
- Precision@K = real_anomalies / K

#### C. Consistency Checks
- Anomaly rate per country (too many = overfitting)
- Clustered anomalies during known global events (COVID, war, oil crisis)?

#### D. Synthetic Anomaly Injection (Optional)
- Inject CPI spikes at random timepoints.
- Confirm model detects them.

---

### 6. 📊 Visualization (Optional But Recommended)
- Line plots of CPI + anomaly markers
- Heatmap of anomaly scores (countries × time)
- Error distributions before and after thresholding

---

### 7. 🚀 Stretch Goals
- Build dashboard with `Plotly Dash` or `Streamlit`
- Add model retraining pipeline (monthly update)
- Use ChatGPT for automatic news summarization per anomaly

---

## 🔁 Remember

Once you finish the model and get a list of anomalies, **send them to me** in JSON/CSV format like:

```json
[
  {"country": "Turkey", "date": "2022-11", "score": 5.31},
  {"country": "Lebanon", "date": "2020-07", "score": 4.85}
]
```

I’ll validate them against real-world news/events and tell you if your model is catching legit crises or just chasing ghosts.

---
