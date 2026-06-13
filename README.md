# Deforestation Detection — Forest / Non-Forest Classification

**Machine Learning Pipeline · Sentinel-2 Spectral Indices (proxy)**

\---

## Overview

This project implements an end-to-end ML pipeline for binary forest cover classification using spectral indices derived from Sentinel-2 satellite imagery. It serves as a methodological prototype demonstrating the feasibility of automated deforestation detection, ahead of applying the same pipeline to real Sentinel-2/Landsat imagery in a Master's research context.

**Author:** Komi Isaac Junior HOUNBO  
**Target programs:** FURG/PPGComp · UTFPR/PPGCTA

\---

## Results

|Model|AUC-ROC (5-Fold CV)|Accuracy (test)|F1-Score (test)|
|-|-:|-:|-:|
|Logistic Regression|0.9498 ± 0.016|—|—|
|**Random Forest**|**0.9468 ± 0.017**|**96.2%**|**96.1%**|
|Gradient Boosting|0.9427 ± 0.017|—|—|

\---

## Dataset

Synthetic dataset (n=2,000) calibrated on published Sentinel-2 spectral signature values for tropical forest and non-forest land cover classes (Huete et al. 2002; Gao 1996; Tucker 1979).

**Features (7 spectral indices):**

|Feature|Description|
|-|-|
|`NDVI`|Normalized Difference Vegetation Index — primary greenness indicator|
|`EVI`|Enhanced Vegetation Index — corrects for atmospheric and soil effects|
|`NDWI`|Normalized Difference Water Index — moisture/water content|
|`NIR`|Near-Infrared reflectance — high for dense vegetation|
|`RED`|Red band reflectance — low for forest (chlorophyll absorption)|
|`SWIR`|Short-Wave Infrared — canopy structure and moisture|
|`Delta\_NDVI`|Temporal NDVI change (t2 − t1) — key deforestation signal|

5% label noise added to simulate real-world annotation uncertainty.

\---

## Project Structure

```
deforestation\_project/
│
├── data/
│   └── sentinel2\_proxy.csv       # Generated dataset
│
├── figures/
│   ├── 01\_eda\_distributions.png  # Feature distributions by class
│   └── 02\_results\_dashboard.png  # Full results dashboard
│
├── Deforestation\_Classification\_ML.ipynb  # ← Main notebook (run on Google Colab)
├── generate\_data.py              # Dataset generation script
├── train\_evaluate.py             # Training \& evaluation script
└── README.md
```

\---

## How to Run

### Option A — Google Colab (recommended)

1. Upload `Deforestation\_Classification\_ML.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells (`Runtime → Run all`)
3. No installation required — all libraries are pre-installed on Colab

### Option B — Local

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python generate\_data.py
python train\_evaluate.py
```

\---

## Key Findings

* **Delta\_NDVI** (temporal variation) is the most discriminative feature: a sharp NDVI drop between two acquisition dates is the primary signal of a logging event.
* **NDVI + NIR** combination effectively separates dense canopy from agricultural or bare soil areas.
* The **Random Forest** classifier achieves stable performance (low cross-validation variance), making it suitable for operational monitoring pipelines.

\---

## Research Perspective

This prototype demonstrates the methodological approach that will be applied at the Master's level to:

* Real Sentinel-2 Level-2A imagery over the Brazilian Cerrado / Atlantic Forest
* Multi-temporal NDVI series for dynamic change detection (2018–2024)
* Advanced architectures: CNN on image patches, LSTM/Transformer on NDVI time series

\---

## References

* Huete, A. et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. *Remote Sensing of Environment*, 83(1-2), 195–213.
* Gao, B.C. (1996). NDWI — A normalized difference water index for remote sensing of vegetation liquid water. *Remote Sensing of Environment*, 58(3), 257–266.
* Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150.
* Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

