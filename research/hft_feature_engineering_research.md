# HFT Feature Engineering Research for Order Book Microstructure
## Short-Horizon Return Prediction with Weak Signals

Research compiled from academic papers, Kaggle competition solutions, and practitioner resources.

---

## 1. ORDER BOOK IMBALANCE FEATURES

### 1.1 Volume Imbalance (Level 1)
The simplest and most predictive single feature. From Cont, Kukanov, Stoikov (2014):

```
ρ = (V_bid - V_ask) / (V_bid + V_ask)
```

- Range: [-1, +1]. Positive = buying pressure, negative = selling pressure.
- Segmentation approach: Buy-heavy (ρ > 1/3), Neutral, Sell-heavy (ρ < -1/3)
- **This is the single strongest predictor of next-tick price direction.**

### 1.2 Multi-Level Depth Imbalance
Static Order Book Imbalance across N levels:

```
OBI_N = (Σᵢ₌₁ᴺ Q_bid^i - Σᵢ₌₁ᴺ Q_ask^i) / (Σᵢ₌₁ᴺ Q_bid^i + Σᵢ₌₁ᴺ Q_ask^i)
```

Depth-weighted variant with geometric decay:
```
OBI_weighted = Σᵢ αⁱ⁻¹ (Q_bid^i - Q_ask^i) / Σᵢ αⁱ⁻¹ (Q_bid^i + Q_ask^i)
```
Typical weights: [1.00, 0.50, 0.25, 0.125, 0.0625] for 5 levels.

**Relevance to our data:** Our D01-D06 features represent 6 depth levels. Construct imbalance at each level and aggregate with decaying weights.

### 1.3 Order Flow Imbalance (OFI) — Cont et al. (2014)
Unlike static imbalance, OFI captures *changes* in the order book:

```
e_n = I{P_bid_n ≥ P_bid_{n-1}} · q_bid_n
    - I{P_bid_n ≤ P_bid_{n-1}} · q_bid_{n-1}
    - I{P_ask_n ≤ P_ask_{n-1}} · q_ask_n
    + I{P_ask_n ≥ P_ask_{n-1}} · q_ask_{n-1}
```

Three cases for bid side (symmetric for ask):
- **Price increases**: All current volume is new → add q_bid_n
- **Price unchanged**: Net change → q_bid_n - q_bid_{n-1}
- **Price decreases**: Previous orders filled → subtract q_bid_{n-1}

Aggregate over time window: `OFI = Σ e_n`

**Key result:** OFI explains ~65% of short-interval price changes (R²), vs ~32% for trade imbalance alone.

### 1.4 Multi-Level OFI (MLOFI) — Xu, Gould, Howison (2019)
Extends OFI to M depth levels. For each level m:

```
MLOFI_k^m = Σ_{events in window k} e^m(τ_n)
```

Vector form: `MLOFI_k = (MLOFI_k^1, ..., MLOFI_k^M)`

**Regression model:**
```
ΔP = α + Σ_{m=1}^M β^m · MLOFI^m + ε
```

**Critical implementation details:**
- Use **ridge regression** (not OLS) — neighboring levels have correlation >0.7-0.9
- Ridge MLOFI (10 levels) reduces RMSE by **65-75%** for large-tick stocks vs single-level OFI
- **PCA integration**: First PC captures >89% of variance across levels
  ```
  Integrated_MLOFI = (w₁ᵀ · MLOFI) / ||w₁||₁
  ```
  where w₁ is the first principal component

**Relevance to our data:** We have D01-D06, with V (volume) and O (order count) at each level. Construct per-level flow changes and aggregate.

### 1.5 Z-Score Normalization of OFI
Critical for stationarity:
```
OFI_norm = (OFI - OFI_rolling_mean) / sqrt(OFI_rolling_var)
```
Use rolling 5-minute window for mean/variance. Compresses raw OFI from (-50, 50) to (-10, 10).

---

## 2. PRICE-BASED MICROSTRUCTURE FEATURES

### 2.1 Microprice (Stoikov 2018)
Better predictor of short-term price moves than mid-price:

```
Microprice = P_ask × (Q_bid / (Q_bid + Q_ask)) + P_bid × (Q_ask / (Q_bid + Q_ask))
```

Equivalently:
```
Microprice = Mid + (Q_bid - Q_ask) / (Q_bid + Q_ask) × Spread/2
```

**Key insight:** The microprice is a better *martingale* estimate of the true price than the mid-price. The weighted mid-price changes on every imbalance update, making it noisy, but it captures the information-weighted fair value.

### 2.2 Volume-Adjusted Mid Price (VAMP)
At best bid/offer:
```
VAMP_bbo = (P_bid × Q_ask + P_ask × Q_bid) / (Q_bid + Q_ask)
```

Extended to N levels:
```
VAMP_N = (Σᵢ P_bid^i × Q_ask^i + Σᵢ P_ask^i × Q_bid^i) / (Σᵢ Q_bid^i + Σᵢ Q_ask^i)
```

### 2.3 Weighted-Depth Order Book Price (WDOBP)
```
WDOBP = (Σᵢ P_bid^i × Q_bid^i + Σᵢ P_ask^i × Q_ask^i) / (Σᵢ Q_bid^i + Σᵢ Q_ask^i)
```
N defined by fixed aggregate quantity rather than percentage depth.

### 2.4 Effective Prices
```
P_eff_bid^N = Σ(P_bid^i × Q_bid^i) / Σ Q_bid^i
P_eff_ask^N = Σ(P_ask^i × Q_ask^i) / Σ Q_ask^i
```

### 2.5 Spread Features
```
Quoted_Spread = P_ask - P_bid
Relative_Spread = (P_ask - P_bid) / Mid
Log_Spread = log(P_ask) - log(P_bid)
```

**Relevance to our data:** Our P (price) features at different D levels allow construction of all these variants. The microprice deviation from mid-price is a key alpha signal.

---

## 3. KYLE'S LAMBDA AND PRICE IMPACT

### 3.1 Kyle's Lambda
Measures price sensitivity to order flow:
```
r_{i,n} = λ_i · S_{i,n} + ε_{i,t}
```

Where signed square-root dollar volume:
```
S_{i,n} = Σ_k sign(v_{k,n}) · √|v_{k,n}|
```

- λ is inversely proportional to liquidity
- Higher λ → lower liquidity → greater price impact per unit of flow
- **OFI-based λ estimates are superior to trade-based estimates**

**Relevance to our data:** Can estimate λ per exchange/asset using our flow (F) and volume (V) features combined with price changes.

---

## 4. TRADE FLOW TOXICITY (VPIN)

### 4.1 VPIN Construction
Volume-Synchronized Probability of Informed Trading (Easley, Lopez de Prado, O'Hara 2012):

1. **Volume bucketing**: Divide trades into equal-volume windows (not equal-time)
2. **Bulk Volume Classification**: Classify each bucket's buy/sell volume
3. **VPIN calculation**:
```
VPIN = (1/n) × Σ_{τ=1}^n |V_buy^τ - V_sell^τ| / V_bucket
```

- Leading indicator of liquidity-induced volatility
- Detected Flash Crash of 2010 hours before it happened
- Measures "toxicity" of order flow — high VPIN = informed traders present

**Relevance to our data:** Our F (flow/filled orders) features can proxy for buy/sell volume classification across exchanges.

---

## 5. INTERACTION AND COMPOSITE FEATURES

### 5.1 From Optiver "Trading at the Close" Competition (Kaggle)

**Market Urgency** (strongest single engineered feature):
```
market_urgency = price_spread × liquidity_imbalance
```

**Price Pressure:**
```
price_pressure = imbalance_size × (ask_price - bid_price)
```

**Depth Pressure:**
```
depth_pressure = (ask_size - bid_size) × (far_price - near_price)
```

**Matched Imbalance:**
```
matched_imbalance = (imbalance_size - matched_size) / (matched_size + imbalance_size)
```

### 5.2 Triplet Features
Compute ratio of ordered triples across price/size combinations:
```
triplet = (max_val - mid_val) / (mid_val - min_val)
```
Applied to all 3-way combinations of prices and sizes.

### 5.3 Cross-Price Imbalances
For all price pairs (p_i, p_j):
```
cross_imbalance = (p_i - p_j) / (p_i + p_j)
```

### 5.4 Dollar Volume
```
dollar_volume = volume × price
```
More informative than raw volume as it captures actual capital flow.

### 5.5 Spread × Depth
```
liquidity_score = spread × (1 / total_depth)
```
Higher values indicate lower liquidity (wide spread + thin book).

---

## 6. TEMPORAL AND LAG FEATURES

### 6.1 Lag Differences (from Optiver competition winners)
For each feature f, compute:
```
diff_1 = f(t) - f(t-1)     # 1-period change
diff_2 = f(t) - f(t-2)     # 2-period change
diff_3 = f(t) - f(t-3)     # 3-period change
pct_change_1 = f(t) / f(t-1) - 1
```

**Relevance:** We already have _LagT1, _LagT2, _LagT3 for each base feature. Engineer:
- `diff = feature - feature_LagT1` (momentum/change)
- `diff_of_diff = (feature - feature_LagT1) - (feature_LagT1 - feature_LagT2)` (acceleration)
- `ratio = feature / feature_LagT1` (relative change)

### 6.2 Rolling Statistics
Over lags, compute:
```
rolling_mean = mean(f, f_lag1, f_lag2, f_lag3)
rolling_std = std(f, f_lag1, f_lag2, f_lag3)
rolling_skew = skew(f, f_lag1, f_lag2, f_lag3)
rolling_range = max(f, ..., f_lag3) - min(f, ..., f_lag3)
```

### 6.3 Autocorrelation Features
```
autocorr_lag1 = corr(f_t, f_{t-1})  # computed over rolling window
```
Autocorrelation decay rate captures market microstructure regimes.

---

## 7. CROSS-SECTIONAL FEATURES

### 7.1 Cross-Sectional Rank Normalization
**Critical for domain transfer (liquid → illiquid).** Replace raw values with ranks:

```python
# Within each time step, rank across all assets
rank_feature = feature.groupby('time').rank(pct=True)  # [0, 1] range
rank_centered = rank_feature - 0.5  # [-0.5, 0.5] range
```

**Why this helps:**
- Eliminates scale differences between liquid and illiquid assets
- Robust to outliers (kurtosis=48 in target)
- Preserves ordinal relationships that transfer across regimes
- Prevents outliers from dominating

### 7.2 Cross-Sectional Demeaning
```
demeaned_feature = feature - feature.groupby('time').mean()
```
Removes common factors (market-wide effects) to isolate stock-specific signal.

### 7.3 Cross-Sectional Z-Score
```
cs_zscore = (feature - cs_mean) / cs_std
```
Where cs_mean and cs_std are computed within each cross-section (time step).

### 7.4 Relative Strength
```
relative_strength = feature / feature.groupby('time').mean()
```
Shows how each asset compares to cross-sectional average.

### 7.5 Cross-Sectional Quantile
```
quantile_feature = feature.groupby('time').rank(pct=True)
```
Maps to uniform [0,1], then optionally to Gaussian via `scipy.stats.norm.ppf()`.

### 7.6 Cross-Asset Order Book Features
From research on cross-impact of OFI:
- OFI from correlated assets predicts returns (cross-impact)
- Aggregate OFI across related assets/exchanges
- "Shocked cross-sectional OBI enhances near-term return forecasting"

**Relevance to our data:** With 5 exchanges (S01-S05), cross-exchange features are natural:
```
exchange_imbalance = S03_feature - mean(S01..S05 features)
exchange_consensus = mean of imbalance signs across exchanges
```

---

## 8. TECHNIQUES FOR HEAVY-TAILED TARGETS (Kurtosis=48)

### 8.1 Target Transformation
```python
# Winsorization: clip extreme values
target_clipped = target.clip(lower=percentile_1, upper=percentile_99)

# Rank transformation of target
target_rank = target.groupby('time').rank(pct=True)
```

### 8.2 Robust Loss Functions
- **Huber loss**: Quadratic for small errors, linear for large → less sensitive to outliers
  ```
  L_δ(a) = a²/2           if |a| ≤ δ
  L_δ(a) = δ(|a| - δ/2)   if |a| > δ
  ```
- **Quantile regression**: Estimate conditional median instead of mean
- **MAE (L1 loss)**: More robust than MSE for heavy tails

### 8.3 Feature Winsorization
```python
feature_winsorized = feature.clip(lower=pct_1, upper=pct_99)
```
Apply before model training to prevent extreme feature values from dominating.

### 8.4 Rank-Based Features
Convert all features to ranks — inherently robust to outliers and heavy tails. This is especially critical when training on liquid assets (likely narrower distributions) and testing on illiquid assets (potentially wider distributions).

---

## 9. DOMAIN ADAPTATION: LIQUID → ILLIQUID TRANSFER

### 9.1 Key Insight from Research
From "Transfer Learning (Il)liquidity" (2025, arxiv:2512.11731):
- Train on liquid proxy with abundant data
- Fine-tune on illiquid target with sparse data
- **Structural features transfer well** — the general shape/behavior of relationships is preserved
- **Scale-dependent features do NOT transfer** — absolute volume, absolute spread, etc.

### 9.2 Features That Transfer Well Across Liquidity Regimes
1. **Normalized/relative features**: Imbalance ratios (already normalized to [-1,1])
2. **Rank-based features**: Cross-sectional ranks are invariant to scale
3. **Ratio features**: Feature ratios are scale-free
4. **Directional indicators**: Signs of changes rather than magnitudes
5. **Microstructural invariants**: Spread relative to tick, depth ratios, imbalance patterns

### 9.3 Features That DO NOT Transfer
1. Raw volumes (differ by orders of magnitude)
2. Raw dollar amounts
3. Absolute spread values
4. Absolute depth values
5. Any feature with strong dependence on trading frequency

### 9.4 Practical Recommendations
- **Always normalize within cross-section** — features become relative
- **Use ratio features** rather than level features
- **Cross-sectional demeaning** removes market-wide effects that may differ between liquid/illiquid
- **Stylized facts of LOBs are universal** across stocks/markets (from microstructure literature)

---

## 10. FEATURE ENGINEERING SPECIFIC TO OUR DATASET

### 10.1 Mapping Our Feature Taxonomy to Literature Features

| Our Feature Type | Literature Equivalent | Engineering Approach |
|---|---|---|
| V (Volume) at D01-D06 | LOB depth volumes | Multi-level imbalance, MLOFI, depth ratios |
| O (Order count) at D01-D06 | Order arrival rates | Order flow, arrival rate imbalance |
| F (Flow/filled) | Trade flow | OFI, VPIN proxy, Kyle's lambda |
| P (Price level) | LOB prices | Microprice, VAMP, spread features |
| B00-B10 / E00-E11 (histograms) | Liquidity distribution | Distribution shape features, entropy |
| A (Alpha/ask signals) | Ask-side specific features | Asymmetric imbalance features |
| W (VWAP/weighted) | VWAP variants | Deviation from VWAP as signal |
| T (Time-weighted) | Time-weighted measures | TWAP deviation, time-vs-volume weighting |
| _LagT1-T3 | Temporal lags | Differences, acceleration, rolling stats |

### 10.2 Highest-Priority Feature Constructions

**Tier 1 — Strongest signal expected (from literature):**
1. **Volume imbalance at D01**: `(V_bid_D01 - V_ask_D01) / (V_bid_D01 + V_ask_D01)`
2. **Multi-level depth imbalance**: Weighted sum across D01-D06
3. **Lag differences**: `feature - feature_LagT1` for key features
4. **Market urgency**: `spread × imbalance`
5. **Cross-sectional rank** of all key features

**Tier 2 — Moderate signal expected:**
6. **Microprice deviation**: `microprice - midprice` (or proxy from our features)
7. **MLOFI PCA**: First PC of multi-level flow imbalances
8. **Depth slope**: How volume changes from D01 to D06 (book shape)
9. **Cross-exchange consensus**: Agreement of imbalance signs across S01-S05
10. **Acceleration**: `diff - diff_lag1` (second derivative of features)

**Tier 3 — Useful but secondary:**
11. **Histogram entropy**: Entropy of B00-B10 / E00-E11 bins
12. **Histogram skewness**: Asymmetry of liquidity distribution
13. **VWAP deviation**: W feature vs P feature differences
14. **Time vs volume weighting gap**: T feature vs W feature differences
15. **Volatility proxy**: Rolling std of lag differences

### 10.3 Cross-Sectional Feature Template
For every raw feature f, construct:
```python
cs_rank = f.groupby('time').rank(pct=True) - 0.5      # rank in [-0.5, 0.5]
cs_demean = f - f.groupby('time').transform('mean')    # demeaned
cs_zscore = (f - cs_mean) / (cs_std + eps)             # z-scored
```

### 10.4 Interaction Feature Template
For pairs of features (a, b):
```python
product = a * b                      # interaction
ratio = a / (b + eps)                # ratio
imbalance = (a - b) / (a + b + eps)  # normalized difference
```

---

## 11. MODEL CONSIDERATIONS WITH WEAK SIGNALS

### 11.1 From Literature & Competition Experience
- **Signal IC ~0.03-0.04 is typical** for microstructure signals after costs
- **Feature selection matters more than model complexity** with weak signals
- Top Kaggle performers: ~70% attributed success to feature engineering, not model choice
- **LightGBM** is dominant in similar competitions (Optiver, Jane Street)
  - 6,300-7,000 estimators typical
  - Purged k-fold cross-validation for temporal data

### 11.2 Feature Selection for Weak Signals
- Use **permutation importance** rather than gain-based importance
- **Mutual information** can capture non-linear relationships
- Start with known-good features from literature (imbalance, OFI, microprice)
- **Fewer, better features** beats many noisy features with weak signals
- Ridge/LASSO for linear models to handle multicollinearity (especially multi-level features)

### 11.3 Ensemble Approach
From Optiver 1st place:
- Multiple LightGBM models with different feature subsets
- Blend predictions (simple averaging or learned weights)
- Each model can focus on different aspect of microstructure

---

## 12. KEY ACADEMIC REFERENCES

1. **Cont, Kukanov, Stoikov (2014)** — "The Price Dynamics of Common Trading Strategies" — OFI framework
2. **Xu, Gould, Howison (2019)** — "Multi-Level Order-Flow Imbalance in a Limit Order Book" — MLOFI
3. **Stoikov (2018)** — "The Micro-Price" — Microprice estimator
4. **Ntakaris et al. (2019)** — "Feature Engineering for Mid-Price Prediction with Deep Learning" — 270+ LOB features
5. **Easley, Lopez de Prado, O'Hara (2012)** — "Flow Toxicity and Liquidity" — VPIN
6. **Kyle (1985)** — Price impact and lambda
7. **Gould, Bonart (2016)** — "Queue Imbalance as a One-Tick-Ahead Price Predictor"
8. **Deep Limit Order Book Forecasting** (2024) — Microstructural guide, LOB representation

---

## SOURCES

- [Order Flow Imbalance - Dean Markwick](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html)
- [Deep Limit Order Book Forecasting](https://arxiv.org/html/2403.09267v1)
- [Market Making with Alpha - Order Book Imbalance](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20Order%20Book%20Imbalance.html)
- [Multi-Level Order-Flow Imbalance](https://www.emergentmind.com/topics/multi-level-order-flow-imbalance-mlofi)
- [Order Book Imbalance in High-Frequency Markets](https://www.emergentmind.com/topics/order-book-imbalance-obi)
- [Kyle's Lambda - frds](https://frds.io/measures/kyle_lambda/)
- [VPIN - The Micro-Price](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)
- [VPIN Explained](https://medium.com/@kryptonlabs/vpin-the-coolest-market-metric-youve-never-heard-of-e7b3d6cbacf1)
- [Key Insights: LOB Imbalance](https://osquant.com/papers/key-insights-limit-order-book/)
- [Ntakaris et al. Feature Engineering](https://arxiv.org/abs/1904.05384)
- [Mid-Price Prediction with Technical Indicators](https://pmc.ncbi.nlm.nih.gov/articles/PMC7292367/)
- [LOB Feature Analysis GitHub](https://github.com/nicolezattarin/LOB-feature-analysis)
- [Optiver Trading at the Close Approach](https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/)
- [Optiver Realized Volatility 1st Place](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970)
- [Transfer Learning (Il)liquidity](https://arxiv.org/abs/2512.11731)
- [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [Hidden Risks of High Kurtosis in ML](https://aicompetence.org/the-hidden-risks-of-high-kurtosis-in-ml/)
- [Cross-Sectional Systematic Strategies via Learning to Rank](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3751012_code4522156.pdf?abstractid=3751012)
- [Queue Imbalance as Price Predictor](https://arxiv.org/pdf/1512.03492)
- [Order Book Filtration and Signal Extraction](https://arxiv.org/html/2507.22712v1)
