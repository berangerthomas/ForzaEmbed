# Guide Pédagogique des Métriques d'Évaluation de Similarité d'Embeddings

## Introduction : Pourquoi Évaluer sans Vérité Terrain ?

Imaginez que vous êtes un bibliothécaire qui doit classer des livres par thème, mais sans avoir de catalogue de référence. Vous devez vous fier à la cohérence de votre système de classification. C'est exactement notre situation avec les embeddings : nous devons évaluer la qualité de nos mesures de similarité en nous appuyant sur des propriétés intrinsèques du système.

**Notre contexte** : Vous avez des embeddings de thèmes de référence et d'un document. Vous voulez savoir si votre mesure de similarité est fiable, sans connaître les "bonnes" réponses à l'avance.

---

## 1. COHÉRENCE INTRA-THÉMATIQUE (ICS)

### 🎯 Définition Simple
La cohérence intra-thématique mesure si votre système donne des scores de similarité "stables" et "prévisibles". Un bon système ne devrait pas donner des scores complètement aléatoires pour des éléments similaires.

### 📐 Formule Mathématique
```
ICS = (1/n) × Σᵢ₌₁ⁿ [σ²(sim(eᵢ, E_doc)) / μ(sim(eᵢ, E_doc))]
```

### 🔍 Interprétation Intuitive
**Pensez à un thermomètre** : 
- Un bon thermomètre donne des mesures cohérentes (faible variance)
- Un mauvais thermomètre oscille de façon imprévisible (forte variance)

**Dans notre cas** :
- **ICS faible (< 0.1)** : Votre système est cohérent, les scores varient peu
- **ICS élevé (> 0.5)** : Votre système est instable, les scores sont erratiques

### 💡 Application Pratique

**Exemple concret** : Supposons que vous comparez un thème "Sport" avec 5 phrases d'un document :

```python
# Scores de similarité obtenus
scores_sport = [0.85, 0.82, 0.87, 0.84, 0.86]  # Cohérent !
scores_incohérent = [0.2, 0.9, 0.1, 0.95, 0.3]  # Problématique !

# Calcul ICS
def calculate_ics(scores):
    mean_score = np.mean(scores)
    variance = np.var(scores)
    return variance / mean_score

ics_bon = calculate_ics(scores_sport)      # ≈ 0.0003 (excellent)
ics_mauvais = calculate_ics(scores_incohérent)  # ≈ 0.64 (médiocre)
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence dans votre cas** : Si votre système de similarité fonctionne bien, il devrait donner des scores relativement stables quand il compare un thème cohérent avec différentes parties d'un document. Cette métrique détecte les dysfonctionnements du système.

---

## 2. SILHOUETTE SCORE MODIFIÉE (SSM)

### 🎯 Définition Simple
La silhouette mesure si les éléments similaires sont effectivement regroupés ensemble et si les éléments différents sont bien séparés. C'est comme évaluer si les invités d'une fête se regroupent naturellement par centres d'intérêt.

### 📐 Formule Mathématique
```
SSM = (1/n) × Σᵢ₌₁ⁿ [(bᵢ - aᵢ) / max(aᵢ, bᵢ)]
```
Où :
- `aᵢ` = distance moyenne aux éléments du même groupe
- `bᵢ` = distance moyenne aux éléments du groupe le plus proche

### 🔍 Interprétation Intuitive
**Analogie du parking** :
- Vous voulez que les voitures de même type se garent ensemble
- `aᵢ` : distance moyenne aux voitures du même type (doit être petite)
- `bᵢ` : distance aux autres types (doit être grande)
- Score élevé : bonne séparation entre types

**Échelle d'interprétation** :
- **+1** : Parfaitement classé (très loin des autres groupes, très proche du sien)
- **0** : À la frontière entre groupes (ambigu)
- **-1** : Mal classé (plus proche des autres groupes que du sien)

### 💡 Application Pratique

**Exemple visuel** : Imaginez 3 thèmes (Sport, Cuisine, Technologie) et leurs distances :

```python
# Pour un embedding du thème "Sport"
distances_meme_theme = [0.1, 0.15, 0.12]  # Autres embeddings sport
distances_autres_themes = [0.7, 0.8, 0.65]  # Cuisine, Techno

a_i = np.mean(distances_meme_theme)    # 0.123
b_i = np.mean(distances_autres_themes) # 0.717

silhouette_score = (b_i - a_i) / max(a_i, b_i)  # (0.717-0.123)/0.717 ≈ 0.83
```

**Score de 0.83** : Excellent ! L'embedding "Sport" est bien plus proche de son propre thème que des autres.

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Dans votre cas, cette métrique vérifie si votre système de similarité sait distinguer les thèmes. Si deux thèmes sont vraiment différents, leurs embeddings devraient être moins similaires entre eux qu'à l'intérieur de chaque thème.

---

## 3. STABILITÉ ANGULAIRE (AS)

### 🎯 Définition Simple
La stabilité angulaire mesure si vos thèmes de référence "pointent" dans la même direction générale que votre document. C'est comme vérifier si plusieurs boussoles indiquent approximativement la même direction.

### 📐 Formule Mathématique
```
AS = (1/|E_ref|) × Σᵢ cos⁻¹(eᵢ · centroïde(E_doc) / (||eᵢ|| × ||centroïde||))
```

### 🔍 Interprétation Intuitive
**Analogie des vecteurs** :
- Chaque embedding est un vecteur dans l'espace multidimensionnel
- L'angle entre deux vecteurs : 0° = identiques, 90° = orthogonaux, 180° = opposés
- Le centroïde du document représente sa "direction moyenne"

**Interprétation des angles** :
- **0-30°** : Excellente alignement (thèmes très cohérents avec le document)
- **30-60°** : Bon alignement (thèmes relativement cohérents)
- **60-90°** : Alignement faible (thèmes peu liés au document)
- **> 90°** : Opposition (thèmes contradictoires avec le document)

### 💡 Application Pratique

**Exemple concret** :

```python
import numpy as np

# Embeddings des thèmes (simplifiés en 2D pour visualisation)
theme_sport = np.array([0.8, 0.6])      # Angle ≈ 37°
theme_cuisine = np.array([0.6, 0.8])    # Angle ≈ 53°
theme_politique = np.array([-0.7, 0.7]) # Angle ≈ 135° (opposé!)

# Centroïde du document
doc_centroid = np.array([1.0, 0.5])     # Direction générale

def calculate_angle(vec1, vec2):
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

angles = [
    calculate_angle(theme_sport, doc_centroid),     # ~37°
    calculate_angle(theme_cuisine, doc_centroid),   # ~53°  
    calculate_angle(theme_politique, doc_centroid)  # ~135°
]

stability_score = np.mean(angles)  # ~75° - alignement moyen
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Si vos thèmes de référence sont pertinents pour le document, ils devraient "pointer" dans des directions similaires dans l'espace des embeddings. Cette métrique détecte les thèmes hors-sujet ou les problèmes d'encodage.

---

## 4. ROBUSTESSE AUX PERTURBATIONS (RS)

### 🎯 Définition Simple
La robustesse teste si votre système reste stable quand on ajoute un petit "bruit" aux données. C'est comme tester si une balance donne le même poids quand on la secoue légèrement.

### 📐 Formule Mathématique
```
RS = 1 - |S(E_ref, E_doc) - S(E_ref + ε, E_doc)| / S(E_ref, E_doc)
```
Où `ε` est un bruit gaussien : `ε ~ N(0, σ²I)`

### 🔍 Interprétation Intuitive
**Analogie de la photo** :
- Une bonne photo reste reconnaissable avec un peu de flou
- Un système robuste donne des scores similaires avec de petites perturbations
- Un système fragile change complètement ses scores au moindre bruit

**Échelle d'interprétation** :
- **RS > 0.95** : Très robuste (scores presque identiques malgré le bruit)
- **RS = 0.80-0.95** : Acceptable (légères variations)
- **RS < 0.80** : Fragile (scores instables)

### 💡 Application Pratique

**Exemple de test** :

```python
# Scores originaux
original_similarities = [0.85, 0.72, 0.91, 0.68]

# Ajout de bruit gaussien (σ = 0.01)
noise = np.random.normal(0, 0.01, size=embedding_dim)
perturbed_embeddings = original_embeddings + noise

# Nouveaux scores
perturbed_similarities = [0.84, 0.73, 0.90, 0.69]

# Calcul de robustesse
def calculate_robustness(original, perturbed):
    differences = np.abs(np.array(original) - np.array(perturbed))
    relative_changes = differences / np.array(original)
    return 1 - np.mean(relative_changes)

robustness = calculate_robustness(original_similarities, perturbed_similarities)
# ≈ 0.97 (excellent - variations < 3%)
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Dans la réalité, vos embeddings peuvent contenir du "bruit" (erreurs d'encodage, variations de modèles, etc.). Un bon système de similarité doit être stable face à ces petites imperfections.

---

## 5. BOOTSTRAP DE CONFIANCE

### 🎯 Définition Simple
Le bootstrap estime la "confiance" que vous pouvez avoir en vos résultats en simulant de nombreuses expériences. C'est comme faire un sondage plusieurs fois pour voir si les résultats sont stables.

### 📐 Principe Mathématique
1. **Rééchantillonnage** : Tirage aléatoire avec remise de vos données
2. **Répétition** : 1000+ fois pour créer une distribution
3. **Intervalle de confiance** : Quantiles 2.5% et 97.5% pour IC à 95%

### 🔍 Interprétation Intuitive
**Analogie du sondage électoral** :
- Un sondage donne 52% d'intentions de vote
- Refait 1000 fois, on obtient des résultats entre 49% et 55%
- L'intervalle [49%, 55%] est l'IC à 95%
- Plus l'intervalle est étroit, plus on est confiant

**Dans notre contexte** :
- Score moyen = 0.75, IC = [0.72, 0.78] → Bon (intervalle étroit)
- Score moyen = 0.75, IC = [0.45, 0.95] → Problématique (très incertain)

### 💡 Application Pratique

**Simulation bootstrap** :

```python
def bootstrap_confidence(similarities, n_iterations=1000):
    bootstrap_means = []
    
    for _ in range(n_iterations):
        # Échantillonnage avec remise
        sample = np.random.choice(similarities, size=len(similarities), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calcul de l'intervalle de confiance à 95%
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'mean': np.mean(similarities),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'uncertainty': ci_upper - ci_lower
    }

# Exemple avec vos scores
similarities = [0.85, 0.72, 0.91, 0.68, 0.79, 0.83]
result = bootstrap_confidence(similarities)

# Résultat possible :
# mean: 0.797, CI: [0.723, 0.867], incertitude: 0.144
```

**Interprétation** : Votre score moyen est 0.797, mais il pourrait raisonnablement varier entre 0.723 et 0.867. L'incertitude de 0.144 indique une précision modérée.

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Cette métrique vous donne une mesure de l'incertitude de vos résultats. Elle est cruciale pour savoir si vos conclusions sont fiables ou si vous avez besoin de plus de données.

---

## 6. DENSITÉ LOCALE (LDI)

### 🎯 Définition Simple
La densité locale mesure si les éléments similaires se "regroupent" naturellement dans l'espace des embeddings. C'est comme observer si les gens ayant les mêmes goûts se retrouvent naturellement dans les mêmes quartiers d'une ville.

### 📐 Formule Mathématique
```
LDI = Σᵢ₌₁ⁿ [|N_k(eᵢ) ∩ Thème(eᵢ)|] / [k × n]
```
Où `N_k(eᵢ)` = les k plus proches voisins de l'embedding `eᵢ`

### 🔍 Interprétation Intuitive
**Analogie du quartier** :
- Vous regardez les 5 plus proches voisins de chaque maison
- Si 4/5 voisins partagent le même "thème" (profession, âge, etc.), c'est bien
- LDI = proportion moyenne de voisins du même thème

**Échelle d'interprétation** :
- **LDI = 1.0** : Parfait (tous les voisins sont du même thème)
- **LDI = 0.8** : Bon (80% des voisins sont cohérents)
- **LDI = 0.2** : Problématique (distribution quasi-aléatoire)

### 💡 Application Pratique

**Exemple de calcul** :

```python
from sklearn.neighbors import NearestNeighbors

def calculate_ldi(embeddings, theme_labels, k=5):
    # Trouver les k plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 car inclut l'élément lui-même
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    total_same_theme = 0
    total_neighbors = 0
    
    for i, neighbors in enumerate(indices):
        current_theme = theme_labels[i]
        # Exclure l'élément lui-même (premier dans la liste)
        neighbor_themes = [theme_labels[j] for j in neighbors[1:]]
        
        same_theme_count = sum(1 for theme in neighbor_themes if theme == current_theme)
        total_same_theme += same_theme_count
        total_neighbors += k
    
    return total_same_theme / total_neighbors

# Exemple
embeddings = [...] # Vos embeddings
themes = ['sport', 'sport', 'cuisine', 'sport', 'cuisine', 'tech', ...]

ldi_score = calculate_ldi(embeddings, themes, k=3)
# Score possible : 0.73 (73% des voisins partagent le même thème)
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Cette métrique vérifie si votre espace d'embeddings a une structure sensée. Si les éléments similaires sont proches et les différents sont éloignés, le LDI sera élevé.

---

## 7. VALIDATION CROISÉE PAR BLOCS (CVIB)

### 🎯 Définition Simple
La validation croisée par blocs teste si vos résultats restent cohérents quand vous divisez vos données en plusieurs groupes. C'est comme vérifier qu'un sondage donne des résultats similaires dans différentes régions.

### 📐 Formule Mathématique
```
CVIB = σ(scores_blocs) / μ(scores_blocs)
```
Coefficient de variation = écart-type / moyenne

### 🔍 Interprétation Intuitive
**Analogie de l'examen** :
- Vous divisez un examen en 4 parties
- Chaque partie devrait donner une note similaire pour un bon étudiant
- Si les notes varient énormément, soit l'examen est mal fait, soit l'étudiant est incohérent

**Interprétation des valeurs** :
- **CVIB < 0.1** : Très stable (variations < 10%)
- **CVIB = 0.1-0.3** : Acceptable (variations modérées)
- **CVIB > 0.5** : Instable (résultats peu fiables)

### 💡 Application Pratique

**Procédure de validation** :

```python
def cross_validation_blocks(ref_embeddings, doc_embeddings, n_blocks=5):
    # Diviser les références en blocs
    block_size = len(ref_embeddings) // n_blocks
    block_scores = []
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size if i < n_blocks-1 else len(ref_embeddings)
        
        # Bloc actuel
        block_refs = ref_embeddings[start_idx:end_idx]
        
        # Calculer similarités pour ce bloc
        similarities = compute_similarities(block_refs, doc_embeddings)
        block_score = np.mean(similarities)
        block_scores.append(block_score)
    
    # Coefficient de variation
    mean_score = np.mean(block_scores)
    std_score = np.std(block_scores)
    cvib = std_score / mean_score
    
    return {
        'block_scores': block_scores,
        'mean': mean_score,
        'std': std_score,
        'cvib': cvib
    }

# Exemple
result = cross_validation_blocks(your_ref_embeddings, your_doc_embeddings)
# Résultat possible :
# block_scores: [0.78, 0.82, 0.75, 0.80, 0.79]
# mean: 0.788, std: 0.025, cvib: 0.032 (très stable !)
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Si votre système de similarité est fiable, il devrait donner des résultats cohérents même en ne regardant qu'une partie de vos thèmes de référence. Cette métrique détecte les biais ou instabilités cachés.

---

## 8. DISTANCE DE WASSERSTEIN

### 🎯 Définition Simple
La distance de Wasserstein mesure "l'effort" nécessaire pour transformer une distribution de scores en une autre. C'est comme calculer le coût minimal pour déménager des tas de sable d'une configuration à une autre.

### 📐 Principe Mathématique
La distance de Wasserstein (ou "Earth Mover's Distance") calcule le coût optimal de transport entre deux distributions :
```
W₁(P, Q) = inf[γ] ∫ ||x - y|| dγ(x,y)
```

### 🔍 Interprétation Intuitive
**Analogie du déménagement** :
- Distribution P : où sont vos meubles actuellement
- Distribution Q : où vous voulez les mettre
- Distance de Wasserstein : coût minimal du déménagement
- Plus c'est cher, plus les distributions sont différentes

**Dans notre contexte** :
- Comparer les distributions de similarités entre différentes métriques
- Distance faible : les métriques donnent des résultats similaires
- Distance élevée : désaccord entre métriques (problème potentiel)

### 💡 Application Pratique

**Comparaison de métriques** :

```python
from scipy.stats import wasserstein_distance

# Scores de deux métriques différentes
scores_cosine = [0.85, 0.72, 0.91, 0.68, 0.79]
scores_euclidean = [0.78, 0.69, 0.88, 0.71, 0.82]

# Distance de Wasserstein
wd = wasserstein_distance(scores_cosine, scores_euclidean)
# Résultat : 0.048

# Interprétation
if wd < 0.1:
    print("Métriques cohérentes")
elif wd < 0.3:
    print("Différences modérées")
else:
    print("Métriques en désaccord")
```

### ✅ Pourquoi Cette Métrique ?
**Pertinence** : Elle permet de comparer différentes métriques de similarité et de détecter si elles sont en accord. Si plusieurs métriques donnent des distributions très différentes, cela signale un problème.

---

## 9. SYNTHÈSE : COMMENT UTILISER CES MÉTRIQUES ENSEMBLE

### 🎯 Stratégie d'Évaluation Globale

**Étape 1 : Cohérence de Base**
- Calculez ICS et Stabilité Angulaire
- Objectif : Vérifier que votre système n'est pas chaotique

**Étape 2 : Structure des Données**
- Calculez SSM et LDI
- Objectif : Vérifier que vos données ont une structure sensée

**Étape 3 : Fiabilité**
- Bootstrap et Validation Croisée
- Objectif : Mesurer l'incertitude de vos résultats

**Étape 4 : Robustesse**
- Test de perturbations
- Objectif : Vérifier la stabilité face aux variations

### 💡 Score Composite Recommandé

```python
def compute_quality_score(metrics):
    weights = {
        'coherence': 0.25,      # ICS inversé et normalisé
        'separation': 0.25,     # Silhouette Score
        'stability': 0.20,      # Bootstrap (incertitude inversée)
        'robustness': 0.20,     # Test de perturbation
        'structure': 0.10       # Densité locale
    }
    
    # Normalisation et inversion si nécessaire
    normalized_metrics = normalize_all_metrics(metrics)
    
    composite_score = sum(weights[key] * normalized_metrics[key] 
                         for key in weights.keys())
    
    return composite_score
```

### 🎖️ Interprétation du Score Final
- **Score > 0.8** : Excellente qualité (système très fiable)
- **Score 0.6-0.8** : Bonne qualité (système acceptable)
- **Score 0.4-0.6** : Qualité modérée (améliorations recommandées)
- **Score < 0.4** : Qualité insuffisante (révision nécessaire)

---

## Conclusion Pratique

Ces métriques forment un ensemble complémentaire pour évaluer votre système de similarité :

1. **Cohérence** : Votre système est-il stable ?
2. **Séparation** : Distingue-t-il bien les différents thèmes ?
3. **Structure** : Les données ont-elles une organisation logique ?
4. **Fiabilité** : Pouvez-vous faire confiance aux résultats ?
5. **Robustesse** : Le système résiste-t-il aux perturbations ?

En combinant ces approches, vous obtenez une évaluation complète et fiable de la qualité de vos calculs de similarité, même sans vérité terrain.