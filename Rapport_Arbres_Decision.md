# Rapport de Projet : Arbres de Décision, Extensions et Applications

**Année universitaire :** 2024-2025  
**Auteur :** Projet Data Mining  
**Date :** 31 Décembre 2025

---

## Table des Matières

1. [Introduction](#introduction)
2. [Technologies Utilisées](#technologies-utilisées)
3. [Partie 1 : Concepts Théoriques](#partie-1--concepts-théoriques)
4. [Partie 2 : Implémentation From Scratch](#partie-2--implémentation-from-scratch)
5. [Partie 3 : Extensions et Expérimentations](#partie-3--extensions-et-expérimentations)
6. [Partie 4 : Application Métier](#partie-4--application-métier)
7. [Conclusion](#conclusion)

---

## Introduction

Ce projet explore en profondeur les **arbres de décision**, une méthode fondamentale de l'apprentissage automatique supervisé. Il combine théorie, implémentation pratique et application réelle sur un cas de diagnostic médical.

### Objectifs du Projet

- Comprendre les fondements mathématiques des arbres de décision
- Implémenter un algorithme d'arbre de décision "from scratch"
- Analyser le phénomène de sur-apprentissage
- Comparer différentes méthodes d'ensemble (Random Forest, AdaBoost)
- Appliquer ces techniques à un problème réel de classification

---

## Technologies Utilisées

| Bibliothèque | Version | Utilisation |
|--------------|---------|-------------|
| **Python** | 3.x | Langage de programmation |
| **NumPy** | - | Calculs numériques et manipulation de tableaux |
| **Pandas** | - | Manipulation et analyse de données |
| **Matplotlib** | - | Visualisation graphique |
| **Scikit-learn** | - | Algorithmes ML de référence |

### Modules Scikit-learn Utilisés

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
```

---

## Partie 1 : Concepts Théoriques

### 1.1 Classification Supervisée

La classification supervisée est une technique d'apprentissage automatique où :

- On dispose d'un ensemble d'exemples **(x, y)** avec **x** les attributs et **y** la classe
- Le but est d'apprendre une fonction **f** telle que **f(x)** prédit **y**
- L'apprentissage est "supervisé" car on connaît les vraies classes

### 1.2 Structure d'un Arbre de Décision

Un arbre de décision est une structure hiérarchique composée de :

| Élément | Description | Exemple |
|---------|-------------|---------|
| **Nœuds internes** | Test sur un attribut | `age > 30 ?` |
| **Branches** | Résultats du test | Oui / Non |
| **Feuilles** | Prédiction de classe | Classe A / Classe B |

### 1.3 Mesures d'Impureté

L'impureté mesure l'hétérogénéité des classes dans un nœud. Trois mesures principales sont étudiées :

#### Indice de Gini

$$Gini(p) = 1 - \sum_{i=1}^{C} p_i^2$$

- Valeur entre 0 (pur) et 0.5 (équilibré pour 2 classes)
- Critère par défaut dans scikit-learn
- Rapide à calculer

#### Entropie

$$H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

- Valeur entre 0 (pur) et 1 (équilibré pour 2 classes)
- Base du gain d'information (Information Gain)
- Utilisé dans l'algorithme ID3/C4.5

#### Erreur de Classification

$$E(p) = 1 - \max(p_i)$$

- Mesure la plus simple
- Moins sensible aux variations de probabilités
- Rarement utilisé en pratique

### 1.4 Implémentation des Fonctions d'Impureté

```python
def gini(counts):
    """Calcule l'indice de Gini pour une distribution de classes."""
    counts = np.array(counts)
    total = np.sum(counts)
    if total == 0:
        return 0
    proportions = counts / total
    return 1 - np.sum(proportions ** 2)

def entropy(counts):
    """Calcule l'entropie pour une distribution de classes."""
    counts = np.array(counts)
    total = np.sum(counts)
    if total == 0:
        return 0
    proportions = counts / total
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))

def classification_error(counts):
    """Calcule l'erreur de classification pour une distribution de classes."""
    counts = np.array(counts)
    total = np.sum(counts)
    if total == 0:
        return 0
    proportions = counts / total
    return 1 - np.max(proportions)
```

### 1.5 Exemples Numériques

| Cas | Distribution | Gini | Entropie | Erreur |
|-----|--------------|------|----------|--------|
| Équilibré | [10, 10] | 0.5000 | 1.0000 | 0.5000 |
| Très pur | [18, 2] | 0.1800 | 0.4690 | 0.1000 |
| Cas 9/1 | [9, 1] | 0.1800 | 0.4690 | 0.1000 |
| Cas 5/5 | [5, 5] | 0.5000 | 1.0000 | 0.5000 |
| Cas 1/9 | [1, 9] | 0.1800 | 0.4690 | 0.1000 |

**Observation clé** : Plus un nœud est pur (dominé par une classe), plus son impureté est faible.

---

## Partie 2 : Implémentation From Scratch

### 2.1 Dataset Pédagogique : Crédit Simplifié

Un dataset fictif pour illustrer le fonctionnement de l'algorithme :

| Propriété | Valeur |
|-----------|--------|
| **Nombre d'exemples** | 12 |
| **Nombre d'attributs** | 4 |
| **Variables** | Propriétaire, État Matrimonial, Revenu, Défaut Précédent |
| **Classes** | Refusé (0), Accordé (1) |

```python
data = np.array([
    [1, 1, 50, 0],  # Proprio, Matrimonial, Revenu, Défaut
    [0, 0, 30, 1],
    [1, 2, 45, 0],
    # ... 12 exemples au total
])
labels = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
```

### 2.2 Structure de Nœud

```python
class Node:
    def __init__(self):
        self.is_leaf = False        # Est-ce une feuille ?
        self.prediction = None      # Classe prédite (si feuille)
        self.feature_index = None   # Index de l'attribut de split
        self.threshold = None       # Seuil de décision
        self.left = None            # Sous-arbre gauche
        self.right = None           # Sous-arbre droit
        self.gini = None            # Impureté du nœud
        self.samples = None         # Nombre d'exemples
```

### 2.3 Fonction de Recherche du Meilleur Split

L'algorithme teste toutes les combinaisons (attribut, seuil) possibles :

**Pour les variables continues :**
- Trie des valeurs uniques
- Test des seuils à mi-chemin entre valeurs consécutives

**Pour les variables catégorielles :**
- Test de chaque valeur unique comme critère de split

```python
def find_best_split(X, y, feature_types):
    # Calcul du Gini parent
    parent_gini = gini(get_class_counts(y))
    
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(n_features):
        # Test de tous les seuils possibles
        for thresh in thresholds:
            # Calcul du gain d'information
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = thresh
    
    return best_feature, best_threshold, best_gain
```

### 2.4 Construction Récursive de l'Arbre

```python
def build_tree(X, y, feature_types, depth=0, max_depth=None, min_samples_leaf=1):
    node = Node()
    
    # Conditions d'arrêt
    if len(np.unique(y)) == 1:           # Nœud pur
        return create_leaf(y)
    if max_depth and depth >= max_depth:  # Profondeur max atteinte
        return create_leaf(y)
    if len(y) < 2 * min_samples_leaf:     # Pas assez d'exemples
        return create_leaf(y)
    
    # Recherche du meilleur split
    best_feature, best_threshold, best_gain = find_best_split(X, y, feature_types)
    
    if best_gain <= 0:
        return create_leaf(y)
    
    # Construction récursive des sous-arbres
    node.left = build_tree(X_left, y_left, ...)
    node.right = build_tree(X_right, y_right, ...)
    
    return node
```

### 2.5 Fonction de Prédiction

```python
def predict_one(node, x, feature_types):
    if node.is_leaf:
        return node.prediction
    
    feature_val = x[node.feature_index]
    
    if feature_types[node.feature_index] == 'continuous':
        if feature_val <= node.threshold:
            return predict_one(node.left, x, feature_types)
        else:
            return predict_one(node.right, x, feature_types)
    else:  # categorical
        if feature_val == node.threshold:
            return predict_one(node.left, x, feature_types)
        else:
            return predict_one(node.right, x, feature_types)
```

### 2.6 Comparaison avec Scikit-learn

| Métrique | Mini-arbre | Scikit-learn |
|----------|------------|--------------|
| Précision entraînement | ~100% | ~100% |
| Temps d'exécution | Plus lent | Optimisé |
| Fonctionnalités | Base | Complètes |

---

## Partie 3 : Extensions et Expérimentations

### 3.1 Dataset Réel : Breast Cancer Wisconsin

| Propriété | Valeur |
|-----------|--------|
| **Nombre d'exemples** | 569 |
| **Nombre d'attributs** | 30 |
| **Classes** | Maligne (0), Bénigne (1) |
| **Distribution** | ~37% Malignes, ~63% Bénignes |

**Attributs** : Caractéristiques cellulaires (rayon, texture, périmètre, aire, lissage, compacité, concavité, symétrie, dimension fractale) - moyennes, erreurs standards, et "pires" valeurs.

### 3.2 Division Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

| Ensemble | Taille |
|----------|--------|
| Train | 398 (70%) |
| Test | 171 (30%) |

### 3.3 Analyse de l'Effet de la Profondeur

| max_depth | Train Acc | Test Acc | Écart |
|-----------|-----------|----------|-------|
| 1 | ~0.91 | ~0.90 | ~0.01 |
| 2 | ~0.94 | ~0.92 | ~0.02 |
| 3 | ~0.96 | ~0.93 | ~0.03 |
| 4 | ~0.98 | ~0.94 | ~0.04 |
| 5 | ~0.99 | ~0.93 | ~0.06 |
| 7 | ~1.00 | ~0.92 | ~0.08 |
| 10 | ~1.00 | ~0.91 | ~0.09 |
| None | ~1.00 | ~0.90 | ~0.10 |

### 3.4 Phénomène de Sur-apprentissage (Overfitting)

**Observations :**

1. Quand `max_depth` augmente → Précision train augmente (jusqu'à 100%)
2. La précision test peut diminuer après un certain seuil
3. Un grand écart entre train et test = **SUR-APPRENTISSAGE**

**Causes :**
- L'arbre "mémorise" les données d'entraînement
- Les règles deviennent trop spécifiques
- Perte de capacité de généralisation

**Solutions :**
- Limiter la profondeur (`max_depth`)
- Imposer un minimum d'échantillons par feuille (`min_samples_leaf`)
- Élagage (pruning)
- Utiliser des méthodes d'ensemble

### 3.5 Random Forest

Random Forest combine plusieurs arbres de décision avec :
- **Bootstrap** : Échantillonnage avec remise
- **Feature sampling** : Sélection aléatoire d'attributs
- **Vote majoritaire** : Agrégation des prédictions

| n_estimators | Train Acc | Test Acc | F1-Score |
|--------------|-----------|----------|----------|
| 10 | ~0.99 | ~0.95 | ~0.96 |
| 50 | ~1.00 | ~0.96 | ~0.97 |
| 100 | ~1.00 | ~0.96 | ~0.97 |
| 200 | ~1.00 | ~0.96 | ~0.97 |

**Avantages :**
- Réduction du sur-apprentissage
- Meilleure généralisation
- Robuste aux outliers
- Estimation de l'importance des variables

### 3.6 AdaBoost (Optionnel)

AdaBoost (Adaptive Boosting) est une méthode de boosting qui :
- Entraîne des classifieurs faibles séquentiellement
- Donne plus de poids aux exemples mal classés
- Combine les classifieurs avec des poids

### 3.7 Comparatif Final

| Méthode | Test Accuracy | Interprétabilité |
|---------|---------------|------------------|
| Arbre simple (max_depth=4) | ~94% | ✅ Élevée |
| Random Forest (n=100) | ~96% | ❌ Faible |
| AdaBoost (n=100) | ~96% | ❌ Faible |

---

## Partie 4 : Application Métier

### 4.1 Domaine d'Application : Diagnostic Médical

**Contexte :** Le diagnostic précoce du cancer du sein est crucial pour le pronostic des patientes. Les médecins utilisent des biopsies pour analyser les caractéristiques des cellules tumorales.

**Objectif :** Classifier automatiquement les tumeurs en :
- **BÉNIGNES** (non cancéreuses) - Classe 1
- **MALIGNES** (cancéreuses) - Classe 0

### 4.2 Modèle Final Retenu

```python
# Arbre de décision optimisé
tree_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)

# Random Forest pour comparaison
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
```

### 4.3 Performances

| Modèle | Accuracy | Précision | Rappel | F1-Score |
|--------|----------|-----------|--------|----------|
| Arbre de décision | ~94% | - | - | - |
| Random Forest | ~96% | - | - | - |

### 4.4 Extraction des Règles de Décision

L'arbre de décision permet d'extraire des règles interprétables :

```
|--- worst radius <= 16.80
|   |--- worst concave points <= 0.14
|   |   |--- class: bénigne
|   |--- worst concave points > 0.14
|   |   |--- class: maligne
|--- worst radius > 16.80
|   |--- class: maligne
```

**Interprétation médicale :**
- Un **rayon maximal > 16.80** indique probablement une tumeur maligne
- Les **points concaves** sont un indicateur clé de malignité

### 4.5 Top 10 Variables les Plus Importantes

| Rang | Variable | Importance |
|------|----------|------------|
| 1 | worst concave points | ~0.15 |
| 2 | worst perimeter | ~0.12 |
| 3 | worst radius | ~0.11 |
| 4 | mean concave points | ~0.10 |
| 5 | worst area | ~0.08 |
| 6 | mean concavity | ~0.07 |
| 7 | mean perimeter | ~0.06 |
| 8 | area error | ~0.05 |
| 9 | worst concavity | ~0.04 |
| 10 | mean radius | ~0.04 |

### 4.6 Matrice de Confusion

```
              Prédit
           Maligne  Bénigne
Réel Maligne   XX      XX    (Vrais Négatifs / Faux Positifs)
     Bénigne   XX      XX    (Faux Négatifs / Vrais Positifs)
```

> **Attention** : Les faux négatifs (malignes classées bénignes) sont particulièrement dangereux en contexte médical !

### 4.7 Discussion

#### Interprétabilité vs Performance

| Critère | Arbre de Décision | Random Forest |
|---------|-------------------|---------------|
| Interprétabilité | ✅ Très bonne | ❌ Faible |
| Performance | ⚠️ Moyenne | ✅ Élevée |
| Sur-apprentissage | ⚠️ Risque élevé | ✅ Risque faible |
| Visualisation | ✅ Facile | ❌ Difficile |
| Explicabilité | ✅ Règles claires | ⚠️ Importance variables |

#### Limites Observées

1. **Zones d'erreur** : Cas limites difficiles à classer
2. **Biais potentiels** : Dataset légèrement déséquilibré
3. **Données manquantes** : Non gérées nativement par les arbres
4. **Généralisation** : Dépend de la population d'entraînement

#### Recommandations pour l'Usage Médical

-  Utiliser l'arbre comme **outil d'aide** au diagnostic, pas de remplacement
- Toujours **confirmer** avec l'expertise médicale
- **Actualiser** régulièrement le modèle avec de nouvelles données
-  Porter une attention particulière aux **faux négatifs**

---

## Conclusion

### Récapitulatif des Apprentissages

Ce projet a permis de maîtriser les aspects suivants :

| Aspect | Compétences Acquises |
|--------|---------------------|
| **Théorie** | Compréhension des mesures d'impureté (Gini, Entropie) |
| **Algorithmique** | Implémentation récursive d'un arbre de décision |
| **Analyse** | Détection et compréhension du sur-apprentissage |
| **Comparaison** | Évaluation de différents modèles (Arbre, RF, AdaBoost) |
| **Application** | Résolution d'un problème réel de diagnostic médical |
| **Interprétation** | Extraction et explication de règles métier |

### Points Clés à Retenir

1. **Les arbres de décision** sont des modèles puissants et interprétables
2. **Le sur-apprentissage** est un risque majeur à contrôler
3. **Les méthodes d'ensemble** (Random Forest) améliorent les performances
4. **L'interprétabilité** est cruciale dans les domaines sensibles comme la médecine
5. **Le compromis biais-variance** guide le choix des hyperparamètres

### Perspectives d'Amélioration

-  Validation croisée (k-fold) pour une évaluation plus robuste
- Optimisation des hyperparamètres (GridSearchCV)
- Analyse des courbes ROC et AUC
- Techniques d'explicabilité avancées (SHAP, LIME)
-  Test sur d'autres datasets médicaux

---

## Annexe : Structure du Code

```
Projet_Arbres_Decision.ipynb
│
├──  Imports et Configuration
│   └── numpy, pandas, matplotlib, sklearn
│
├── Partie 1 : Concepts Théoriques
│   ├── Définitions markdown
│   ├── gini()
│   ├── entropy()
│   ├── classification_error()
│   └── Visualisation des courbes d'impureté
│
├──  Partie 2 : Implementation From Scratch
│   ├── Dataset crédit simplifié
│   ├── class Node
│   ├── get_class_counts()
│   ├── find_best_split()
│   ├── build_tree()
│   ├── predict_one() / predict()
│   ├── print_tree()
│   └── Comparaison sklearn
│
├──  Partie 3 : Extensions
│   ├── Chargement Breast Cancer
│   ├── Train/Test split
│   ├── Analyse profondeur
│   ├── Graphique overfitting
│   ├── Random Forest
│   └── AdaBoost
│
└──  Partie 4 : Application Métier
    ├── Contexte médical
    ├── Modèle final
    ├── Règles de décision
    ├── Importance des variables
    ├── Matrice de confusion
    └── Discussion et recommandations
```

---

*Rapport généré automatiquement - Projet Data Mining 2024-2025*
