
# Lab 1: Classification & Regression Architectures using PyTorch
---

## Vue d'ensemble du Projet
Ce laboratoire vise à se familiariser avec l'écosystème **PyTorch** pour concevoir, entraîner, optimiser et évaluer des architectures de réseaux de neurones profonds (DNN / MLP). Le travail est structuré en deux parties principales :
- **Partie 1 - Régression :** Évaluation de performances de prédiction financière à partir des données d'actions NYSE (`prices-split-adjusted.csv`).
- **Partie 2 - Classification Multi-classe :** Détection et catégorisation de pannes industrielles sur un dataset de maintenance prédictive (`predictive_maintenance.csv`).

---

## Partie 1 : Régression - New York Stock Exchange (NYSE)

### 1. Workflow de la Partie 1

```

[ Dataset Brut NYSE ] 
       │
       ▼
[ 1. Chargement & Sélection ] ──► Importation de 'prices-split-adjusted.csv' et choix de 'close' comme  cible.
       │
       ▼
[ 2. EDA ]► Analyse des distributions et calcul de la matrice de corrélation (mise en évidence du r ≈ 1.00 entre open/high/low et close).
       │
       ▼
[ 3. Train / Test Split ] ──────► Séparation 80% Train / 20% Test des features (open, high, low, volume) et de la target (close).
       │
       ▼
[ 4. Standardisation ] ─────────► Normalisation des features et de la cible via StandardScaler (fit sur Train, transform sur Test).
       │
       ▼
[ 5. Optimisation (GridSearch) ]► Recherche des meilleurs hyperparamètres via skorch (Adam, lr=0.001, batch_size=256).
       │
       ▼
[ 6. Entraînement du MLP ] ─────► Construction de l'architecture MLP (4 -> 64 -> 32 -> 1) avec activation ReLU et MSELoss.
       │
       ▼
[ 7. Évaluation du Modèle ] ────► Visualisation des courbes (Loss MSE vs Époques) et calcul du score R² (R² ≈ 0.9999).
       │
       ▼
[ 8. Test de Régularisation ] ──► Ajout de Dropout (0.05) et Weight Decay (L2 = 1e-3/1e-4) pour analyser l'impact du bruit sur la stabilité de convergence.

```

### 2. Outils & Bibliothèques Utilisés
* **Analyse & Traitement des Données :** `pandas`, `numpy`
* **Visualisation :** `matplotlib`, `seaborn`
* **Prétraitement & Évaluation :** `scikit-learn` (`StandardScaler`, `train_test_split`, `r2_score`, `GridSearchCV`)
* **Modélisation Deep Learning :** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data` (`TensorDataset`, `DataLoader`)
* **Wrapper Sklearn/PyTorch :** `skorch` (`NeuralNetRegressor`)

### 3. Étapes Clés & Analyses
* **Exploratory Data Analysis (EDA) :** Mise en évidence d'une corrélation quasi-parfaite ($r = 1.00$) entre les prix de la même journée (`open`, `high`, `low`) et la variable cible de clôture (`close`). Le volume présente une corrélation beaucoup plus faible ($-0.13$).
* **Architecture DNN :** Construction d'un MLP à 3 couches linéaires ($4 \to 64 \to 32 \to 1$) utilisant l'activation `ReLU` et la perte `MSELoss`.
* **Optimisation par GridSearch :** Sélection des hyperparamètres optimaux via `GridSearchCV` et `skorch` : Optimiseur `Adam`, Learning Rate = $0.001$, Batch Size = $256$, Époques = $10$.
* **Résultats du Modèle de Base :** Convergence très rapide dès la 2ᵉ époque avec une perte MSE de $0.0001$ et un score $R^2 = 0.9999$.
* **Analyse de la Régularisation :** L'ajout d'une régularisation excessive (combinaison `BatchNorm1d` + `Dropout` agressif) sur une relation quasi-linéaire simple crée une instabilité de variance et augmente la perte de test. Une régularisation légère (`Dropout(0.05)` + `weight_decay = 1e-4`) préserve la stabilité et élimine tout risque de surapprentissage.

---

## Partie 2 : Classification Multi-classe - Predictive Maintenance

### 1. Workflow de la Partie 2


```
[ Dataset Brut ] 
       │
       ▼
[ 1. Nettoyage & Filtrage ] ──► Suppression des colonnes non informatives (UDI, Product ID, Target binaire).
       │
       ▼
[ 2. Encodage des Variables ] ──► Transformation de 'Type' (L, M, H -> 0, 1, 2) et 'Failure Type' (LabelEncoder).
       │
       ▼
[ 3. Split Stratifié ] ────────► Séparation 80% Train / 20% Test (en préservant la répartition des pannes).
       │
       ▼
[ 4. Standardisation ] ────────► Normalisation des features via StandardScaler (fit sur Train, transform sur Test).
       │
       ▼
[ 5. Data Augmentation ] ──────► Application de SMOTE uniquement sur Train pour équilibrer les 6 classes.
       │
       ▼
[ 6. Optimisation (GridSearch)]─► Recherche d'hyperparamètres via skorch (Learning Rate, Optimiseur, Batch Size).
       │
       ▼
[ 7. Entraînement du DNN ] ────► Construction de l'architecture MLP PyTorch et calcul de la CrossEntropyLoss.
       │
       ▼
[ 8. Évaluation Complète ] ────► Courbes Loss/Accuracy, Matrice de confusion, F1-Score Macro et Recall.
       │
       ▼
[ 9. Régularisation ] ────────► Test du Dropout + Weight Decay (L2) pour contrer le surapprentissage .
```

### 2. Outils & Bibliothèques Utilisés
* **Analyse & Manipulation :** `pandas`, `numpy`, `collections`
* **Visualisation :** `matplotlib`, `seaborn`
* **Prétraitement & Métriques :** `scikit-learn` (`StandardScaler`, `LabelEncoder`, `train_test_split`, `accuracy_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`)
* **Rééquilibrage des Données :** `imbalanced-learn` (`SMOTE`)
* **Deep Learning & Grille :** `torch`, `torch.nn`, `torch.optim`, `skorch` (`NeuralNetClassifier`)

### 3. Étapes Clés & Analyses
* **Prétraitement :** Suppression des colonnes d'identification non pertinentes (`UDI`, `Product ID`, `Target`). Encodage ordinal de `Type` ($L=0, M=1, H=2$) et conversion de `Failure Type` en 6 labels numériques ($0$ à $5$).
* **Déséquilibre Extrême des Données :** L'EDA a révélé que la classe `No Failure` représente $96.5\%$ des données ($7722$ cas sur $8000$ en Train), tandis que certaines pannes comme `Random Failures` ne comptent que 14 exemples.
* **Rééquilibrage par SMOTE :** Application de l'algorithme SMOTE uniquement sur l'ensemble d'entraînement pour équilibrer chaque classe à $7722$ échantillons (taille totale du train rééquilibré : $46\,332$ lignes).
* **GridSearch & Entraînement :** Meilleurs hyperparamètres retenus : `Adam`, $lr = 0.01$, `batch_size = 256`. L'entraînement du MLP multi-classe ($6 \to 64 \to 32 \to 6$) avec la fonction `CrossEntropyLoss` atteint $94.35\%$ d'accuracy globale sur l'ensemble de test.
* **Analyse des Performances par Métriques :**
  * **Pannes Déterministes :** Excellente détection des pannes d'origine physique (`Power Failure`, `Overstrain Failure`, `Heat Dissipation Failure`) avec un taux de Rappel (Recall) $> 80\%$.
  * **Pannes Atypiques :** Échec sur les `Random Failures` ($0\%$ de recall) en raison de leur nature purement stochastique, et faible précision sur `Tool Wear Failure` due aux faux positifs créés par le suréchantillonnage synthétique.
* **Impact de la Régularisation :** L'intégration du `Dropout(0.2)` et du `Weight Decay` ($10^{-4}$) a permis de lisser la Test Loss et d'arrêter la remontée de l'overfitting observée après la 10ᵉ époque du modèle initial, au prix d'une légère baisse de l'accuracy globale ($89.40\%$).

---

## Synthèse & Synthèse des Apprentissages
Ce laboratoire m'a permis de maîtriser la méthodologie complète de conception de réseaux de neurones profonds sous **PyTorch**, depuis la préparation rigoureuse des données jusqu'à l'évaluation fine des modèles. J'ai notamment appris à quel point un fort déséquilibre de classes peut tromper la métrique d'Accuracy et comment l'utilisation de **SMOTE** associée au **F1-Score Macro** permet de construire des modèles de détection réels et fiables. Enfin, ces expérimentations m'ont démontré qu'appliquer la régularisation n'est pas automatique : elle doit être minutieusement dosée selon la complexité intrinsèque de la fonction à apprendre pour éviter le sous-apprentissage.
