"""
Projet 1 : Classification Forêt / Non-Forêt à partir d'indices spectraux Sentinel-2
======================================================================================
Auteur  : Komi Isaac Junior HOUNBO
Objectif: Prototype de pipeline ML pour la détection de déforestation
          (proxy données multitemporelles Sentinel-2)
Candidatures ciblées : FURG/PPGComp · UTFPR/PPGCTA
"""

# ─── 0. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

SEED = 42
PALETTE = {'Non-Forêt': '#d4720a', 'Forêt': '#2d8a4e'}

# ─── 1. Chargement des données ────────────────────────────────────────────────
df = pd.read_csv('data/sentinel2_proxy.csv')
df['class_name'] = df['label'].map({1: 'Forêt', 0: 'Non-Forêt'})

print("=" * 60)
print("1. APERÇU DES DONNÉES")
print("=" * 60)
print(f"Dimensions : {df.shape}")
print(f"\nDistribution des classes :")
print(df['class_name'].value_counts())
print(f"\nStatistiques descriptives :")
print(df.drop(columns=['label', 'class_name']).describe().round(3))

# ─── 2. Analyse exploratoire (EDA) ──────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Analyse Exploratoire — Distributions des Indices Spectraux Sentinel-2',
             fontsize=14, fontweight='bold', y=1.01)

features = ['NDVI', 'EVI', 'NDWI', 'NIR', 'RED', 'SWIR', 'Delta_NDVI']
colors = [PALETTE['Forêt'], PALETTE['Non-Forêt']]

for i, feat in enumerate(features):
    ax = axes[i // 4][i % 4]
    for cls, color in zip(['Forêt', 'Non-Forêt'], colors):
        subset = df[df['class_name'] == cls][feat]
        ax.hist(subset, bins=35, alpha=0.65, color=color, label=cls, edgecolor='white', linewidth=0.3)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

# Dernière case : matrice de corrélation simplifiée
ax_corr = axes[1][3]
corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax_corr, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
            annot_kws={'size': 7}, mask=mask, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax_corr.set_title('Corrélations entre features', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/01_eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Figure EDA sauvegardée")

# ─── 3. Préparation train/test ────────────────────────────────────────────────
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
print(f"\n{'='*60}")
print("2. SPLIT TRAIN / TEST")
print(f"{'='*60}")
print(f"Train : {X_train.shape[0]} échantillons")
print(f"Test  : {X_test.shape[0]} échantillons")

# ─── 4. Modèles ───────────────────────────────────────────────────────────────
models = {
    'Régression Logistique': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, random_state=SEED))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                       min_samples_leaf=5, random_state=SEED))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                            learning_rate=0.1, random_state=SEED))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}

print(f"\n{'='*60}")
print("3. VALIDATION CROISÉE (5-Fold)")
print(f"{'='*60}")

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    results[name] = scores
    print(f"{name:<30} AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ─── 5. Entraînement & évaluation du meilleur modèle (Random Forest) ─────────
print(f"\n{'='*60}")
print("4. ÉVALUATION DÉTAILLÉE — RANDOM FOREST")
print(f"{'='*60}")

rf_model = models['Random Forest']
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"\nRapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=['Non-Forêt', 'Forêt'], digits=4))

auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC : {auc:.4f}")

# ─── 6. Visualisations finales ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# 6a. Comparaison AUC cross-validation
ax1 = fig.add_subplot(gs[0, 0])
model_names = list(results.keys())
means = [results[m].mean() for m in model_names]
stds  = [results[m].std()  for m in model_names]
colors_bar = ['#5b8dd9', '#2d8a4e', '#c0392b']
bars = ax1.bar(range(len(model_names)), means, yerr=stds, capsize=6,
               color=colors_bar, alpha=0.85, edgecolor='white', linewidth=1.2)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(['Log. Reg.', 'Rnd Forest', 'Grad. Boost'], fontsize=9)
ax1.set_ylabel('AUC-ROC (5-Fold CV)')
ax1.set_title('Comparaison des modèles', fontweight='bold')
ax1.set_ylim(0.5, 1.05)
for bar, m, s in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
             f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)

# 6b. Matrice de confusion
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Non-Forêt', 'Forêt'])
disp.plot(ax=ax2, colorbar=False, cmap='Greens')
ax2.set_title('Matrice de Confusion\n(Random Forest, jeu de test)', fontweight='bold')

# 6c. Courbe ROC
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color='#2d8a4e', lw=2.5, label=f'AUC = {auc:.4f}')
ax3.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Aléatoire')
ax3.set_xlabel('Taux de Faux Positifs')
ax3.set_ylabel('Taux de Vrais Positifs')
ax3.set_title('Courbe ROC — Random Forest', fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.spines[['top', 'right']].set_visible(False)

# 6d. Importance des variables
ax4 = fig.add_subplot(gs[0, 3])
importances = rf_model.named_steps['clf'].feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
colors_imp = ['#c0392b' if 'Delta' in f or f == 'NDVI' else '#2d8a4e'
              for f in feat_imp.index]
feat_imp.plot(kind='barh', ax=ax4, color=colors_imp, edgecolor='white', linewidth=0.5)
ax4.set_title('Importance des variables\n(Gini — Random Forest)', fontweight='bold')
ax4.set_xlabel('Importance relative')
ax4.spines[['top', 'right']].set_visible(False)

# 6e. Scatter NDVI vs Delta_NDVI (plan décision)
ax5 = fig.add_subplot(gs[1, 0:2])
for cls, color, marker in zip([1, 0], [PALETTE['Forêt'], PALETTE['Non-Forêt']], ['o', 's']):
    mask = y_test == cls
    label = 'Forêt' if cls == 1 else 'Non-Forêt'
    ax5.scatter(X_test.loc[y_test == cls, 'NDVI'],
                X_test.loc[y_test == cls, 'Delta_NDVI'],
                c=color, alpha=0.45, s=20, label=label, marker=marker)
ax5.set_xlabel('NDVI (t2)')
ax5.set_ylabel('Δ NDVI (t2 − t1)')
ax5.set_title('Distribution Forêt / Non-Forêt\nNDVI vs Variation temporelle (Δ NDVI)', fontweight='bold')
ax5.legend(fontsize=10)
ax5.spines[['top', 'right']].set_visible(False)
ax5.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

# 6f. Boîtes à moustaches NDVI & EVI
ax6 = fig.add_subplot(gs[1, 2])
df_box = df[['NDVI', 'EVI', 'class_name']].melt(id_vars='class_name',
                                                  var_name='Indice', value_name='Valeur')
sns.boxplot(data=df_box, x='Indice', y='Valeur', hue='class_name',
            palette=PALETTE, ax=ax6, linewidth=1.2, fliersize=3)
ax6.set_title('NDVI & EVI par classe', fontweight='bold')
ax6.legend(title='', fontsize=9)
ax6.spines[['top', 'right']].set_visible(False)

# 6g. Résumé textuel des métriques
ax7 = fig.add_subplot(gs[1, 3])
ax7.axis('off')
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
metrics_text = [
    ('Accuracy',  f"{accuracy_score(y_test, y_pred):.4f}"),
    ('Précision', f"{precision_score(y_test, y_pred):.4f}"),
    ('Rappel',    f"{recall_score(y_test, y_pred):.4f}"),
    ('F1-Score',  f"{f1_score(y_test, y_pred):.4f}"),
    ('AUC-ROC',   f"{auc:.4f}"),
    ('Échantillons train', str(X_train.shape[0])),
    ('Échantillons test',  str(X_test.shape[0])),
    ('Features',  str(len(features))),
]
table = ax7.table(
    cellText=metrics_text,
    colLabels=['Métrique', 'Valeur'],
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2d8a4e')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f7f2')
    cell.set_edgecolor('#cccccc')
ax7.set_title('Résumé des performances\n(Random Forest)', fontweight='bold', pad=10)

fig.suptitle('Détection de Déforestation — Classification Forêt/Non-Forêt\n'
             'Pipeline Machine Learning · Indices Spectraux Sentinel-2 (proxy)',
             fontsize=13, fontweight='bold', y=1.02)

plt.savefig('figures/02_results_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Dashboard résultats sauvegardé")

print(f"\n{'='*60}")
print("5. RÉSUMÉ FINAL")
print(f"{'='*60}")
print(f"Meilleur modèle : Random Forest")
print(f"AUC-ROC (CV 5-Fold) : {results['Random Forest'].mean():.4f}")
print(f"AUC-ROC (test)      : {auc:.4f}")
print(f"Accuracy (test)     : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (test)     : {f1_score(y_test, y_pred):.4f}")
print(f"\nFigures générées dans ./figures/")
