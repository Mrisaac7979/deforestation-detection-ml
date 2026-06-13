"""
Génération d'un dataset synthétique réaliste simulant des indices spectraux Sentinel-2
pour la classification Forêt / Non-Forêt (proxy déforestation).

Les distributions sont calibrées sur des valeurs typiques de la littérature
(Huete et al. 2002, Gao 1996, Tucker 1979).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # 1000 Forêt, 1000 Non-Forêt

# ─── Classe 1 : Forêt ────────────────────────────────────────────────────────
n_forest = N // 2

# NDVI (Normalized Difference Vegetation Index) : forêt dense → 0.55–0.85
ndvi_forest = np.random.normal(0.70, 0.10, n_forest).clip(0.30, 0.95)

# EVI (Enhanced Vegetation Index) : corrélé à NDVI + bruit
evi_forest = 0.45 * ndvi_forest + np.random.normal(0.05, 0.06, n_forest)
evi_forest = evi_forest.clip(0.10, 0.80)

# NDWI (Water Index) : végétation dense → valeurs négatives à légèrement positives
ndwi_forest = np.random.normal(-0.20, 0.12, n_forest).clip(-0.60, 0.20)

# NIR (bande proche infrarouge) : élevé pour forêt
nir_forest = np.random.normal(0.50, 0.08, n_forest).clip(0.25, 0.75)

# RED (bande rouge) : faible pour forêt (absorption chlorophylle)
red_forest = np.random.normal(0.07, 0.025, n_forest).clip(0.02, 0.20)

# SWIR (Short-Wave Infrared) : modéré
swir_forest = np.random.normal(0.18, 0.05, n_forest).clip(0.05, 0.40)

# Variation temporelle NDVI (différence t2-t1) : stable → faible delta
delta_ndvi_forest = np.random.normal(-0.03, 0.06, n_forest).clip(-0.25, 0.15)

# ─── Classe 0 : Non-Forêt (agriculture, zone urbaine, sol nu, eau) ───────────
n_nonforest = N // 2

ndvi_nf = np.random.normal(0.25, 0.12, n_nonforest).clip(-0.10, 0.55)
evi_nf = 0.45 * ndvi_nf + np.random.normal(-0.05, 0.07, n_nonforest)
evi_nf = evi_nf.clip(-0.05, 0.50)
ndwi_nf = np.random.normal(0.10, 0.18, n_nonforest).clip(-0.30, 0.60)
nir_nf = np.random.normal(0.28, 0.10, n_nonforest).clip(0.05, 0.55)
red_nf = np.random.normal(0.15, 0.06, n_nonforest).clip(0.03, 0.40)
swir_nf = np.random.normal(0.32, 0.09, n_nonforest).clip(0.10, 0.60)
delta_ndvi_nf = np.random.normal(-0.18, 0.10, n_nonforest).clip(-0.50, 0.10)

# ─── Assemblage ──────────────────────────────────────────────────────────────
df_forest = pd.DataFrame({
    'NDVI': ndvi_forest, 'EVI': evi_forest, 'NDWI': ndwi_forest,
    'NIR': nir_forest, 'RED': red_forest, 'SWIR': swir_forest,
    'Delta_NDVI': delta_ndvi_forest, 'label': 1
})

df_nf = pd.DataFrame({
    'NDVI': ndvi_nf, 'EVI': evi_nf, 'NDWI': ndwi_nf,
    'NIR': nir_nf, 'RED': red_nf, 'SWIR': swir_nf,
    'Delta_NDVI': delta_ndvi_nf, 'label': 0
})

df = pd.concat([df_forest, df_nf], ignore_index=True)

# Bruit de label (5 % d'erreurs d'annotation — réaliste)
noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('data/sentinel2_proxy.csv', index=False)
print(f"Dataset sauvegardé : {df.shape[0]} échantillons, {df.shape[1]-1} features")
print(df['label'].value_counts().rename({1: 'Forêt', 0: 'Non-Forêt'}))
