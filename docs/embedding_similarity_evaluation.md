# Évaluation de la Précision des Calculs de Similarité d'Embeddings sans Vérité Terrain

## 1. Introduction et Problématique

L'évaluation de la précision des calculs de similarité d'embeddings sans vérité terrain représente un défi majeur en apprentissage automatique. Dans votre cas, vous disposez de :
- **E_ref** = {e₁, e₂, ..., e_n} : embeddings des thèmes de référence
- **E_doc** = {d₁, d₂, ..., d_m} : embeddings du document
- **S** = fonction de similarité (cosinus, euclidienne, etc.)

L'objectif est d'évaluer la qualité de S(E_ref, E_doc) sans connaître les appariements "vrais".

## 2. Métriques Intrinsèques de Cohérence

### 2.1 Cohérence Intra-Thématique

**Principe** : Un bon système de similarité doit produire des scores cohérents entre embeddings similaires.

**Métrique de Cohérence Interne (ICS)**:
```
ICS = (1/n) × Σᵢ₌₁ⁿ [σ²(sim(eᵢ, E_doc)) / μ(sim(eᵢ, E_doc))]
```

Où :
- σ² = variance des scores de similarité
- μ = moyenne des scores
- Plus ICS est faible, plus le système est cohérent

**Calcul pratique** :
```python
def coherence_score(ref_embeddings, doc_embeddings, sim_func):
    coherence_scores = []
    for ref_emb in ref_embeddings:
        similarities = [sim_func(ref_emb, doc_emb) for doc_emb in doc_embeddings]
        mean_sim = np.mean(similarities)
        var_sim = np.var(similarities)
        coherence_scores.append(var_sim / (mean_sim + 1e-8))
    return np.mean(coherence_scores)
```

### 2.2 Métrique de Stabilité Angulaire

**Formule** :
```
AS = (1/|E_ref|) × Σᵢ ∠(eᵢ, centroïde(E_doc))
```

Où ∠ représente l'angle cosinus entre le vecteur de référence et le centroïde des embeddings du document.

## 3. Analyse de Distribution et Séparabilité

### 3.1 Indice de Silhouette Adapté et Décomposé

**Silhouette Score avec Composantes Normalisées**:

Le score silhouette traditionnel est décomposé en ses composantes fondamentales pour une meilleure interprétation :

```
SSM = (1/n) × Σᵢ₌₁ⁿ [(bᵢ - aᵢ) / max(aᵢ, bᵢ)]
```

**Composantes Normalisées** :
- **Distance Intra-Cluster Normalisée** : `1 - (moyenne(aᵢ) / distance_max)`
  - Mesure la qualité de cohésion interne (0-1, plus élevé = meilleur)
  - Valeur proche de 1 : points très proches au sein des clusters
- **Distance Inter-Cluster Normalisée** : `moyenne(bᵢ) / distance_max`
  - Mesure la qualité de séparation (0-1, plus élevé = meilleur)  
  - Valeur proche de 1 : clusters bien séparés

**Implémentation** :
```python
def calculate_silhouette_metrics(embeddings, labels, metric="cosine"):
    # Calcul de la matrice des distances
    distance_matrix = pairwise_distances(embeddings, metric=metric)
    
    a_values = []  # distances intra-cluster
    b_values = []  # distances inter-cluster
    
    for i in range(len(embeddings)):
        current_label = labels[i]
        
        # a(i): distance moyenne intra-cluster
        same_cluster_mask = (labels == current_label) & (np.arange(len(embeddings)) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(distance_matrix[i][same_cluster_mask])
        else:
            a_i = 0.0
        a_values.append(a_i)
        
        # b(i): distance moyenne au cluster le plus proche
        b_i = np.inf
        for other_label in np.unique(labels):
            if other_label != current_label:
                other_cluster_mask = labels == other_label
                if np.sum(other_cluster_mask) > 0:
                    mean_dist = np.mean(distance_matrix[i][other_cluster_mask])
                    b_i = min(b_i, mean_dist)
        b_values.append(b_i if b_i != np.inf else 0.0)
    
    # Normalisation
    max_distance = np.max(distance_matrix)
    intra_normalized = 1 - (np.mean(a_values) / max_distance)
    inter_normalized = np.mean(b_values) / max_distance
    
    # Score silhouette standard
    silhouette = np.mean([(b-a)/max(a,b) for a,b in zip(a_values, b_values) if max(a,b) > 0])
    
    return {
        "intra_cluster_distance_normalized": intra_normalized,
        "inter_cluster_distance_normalized": inter_normalized, 
        "silhouette_score": silhouette
    }
```

### 3.2 Métrique de Densité Locale

**Local Density Index (LDI)** :
```
LDI = Σᵢ₌₁ⁿ [|N_k(eᵢ) ∩ Thème(eᵢ)|] / [k × n]
```

Où N_k(eᵢ) représente les k plus proches voisins de eᵢ.

## 4. Validation Croisée et Bootstrap

### 4.1 Validation Croisée par Blocs

**Méthodologie** :
1. Diviser E_ref en k blocs
2. Pour chaque bloc, calculer la similarité avec E_doc
3. Mesurer la variance inter-blocs

**Coefficient de Variation Inter-Blocs (CVIB)** :
```
CVIB = σ(scores_blocs) / μ(scores_blocs)
```

### 4.2 Bootstrap de Confiance

**Intervalle de Confiance Bootstrap** :
```python
def bootstrap_confidence_interval(similarities, n_bootstrap=1000, confidence=0.95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(similarities, size=len(similarities), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return lower, upper, np.std(bootstrap_means)
```

## 5. Métriques de Robustesse

### 5.1 Test de Perturbation

**Robustness Score (RS)** :
```
RS = 1 - |S(E_ref, E_doc) - S(E_ref + ε, E_doc)| / S(E_ref, E_doc)
```

Où ε est un bruit gaussien : ε ~ N(0, σ²I)

### 5.2 Analyse de Sensibilité

**Gradient de Sensibilité** :
```
∇S = ∂S/∂E_ref = lim[h→0] [S(E_ref + h) - S(E_ref)] / h
```

**Implémentation numérique** :
```python
def sensitivity_analysis(ref_embeddings, doc_embeddings, sim_func, epsilon=1e-6):
    base_similarities = compute_similarities(ref_embeddings, doc_embeddings, sim_func)
    sensitivities = []
    
    for i, ref_emb in enumerate(ref_embeddings):
        # Perturbation dans chaque dimension
        gradients = []
        for dim in range(ref_emb.shape[0]):
            perturbed = ref_emb.copy()
            perturbed[dim] += epsilon
            
            perturbed_similarities = compute_similarities([perturbed], doc_embeddings, sim_func)
            gradient = (perturbed_similarities[0] - base_similarities[i]) / epsilon
            gradients.append(gradient)
        
        sensitivities.append(np.linalg.norm(gradients))
    
    return np.mean(sensitivities)
```

## 6. Approches Comparative et Relative

### 6.1 Comparaison Multi-Métriques

**Score Composite Normalisé (SCN)** :
```
SCN = Σᵢ wᵢ × (mᵢ - min(mᵢ)) / (max(mᵢ) - min(mᵢ))
```

Où mᵢ sont différentes métriques de similarité et wᵢ leurs poids respectifs.

### 6.2 Analyse en Composantes Principales des Similarités

**PCA des Matrices de Similarité** :
```python
def pca_similarity_analysis(similarity_matrices):
    # Aplatir les matrices de similarité
    flattened = [matrix.flatten() for matrix in similarity_matrices]
    
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(flattened)
    
    # Variance expliquée
    explained_variance = pca.explained_variance_ratio_
    
    return components, explained_variance
```

## 7. Métriques Basées sur la Géométrie

### 7.1 Distance de Wasserstein

**Distance de Wasserstein entre Distributions** :
```
W₁(P, Q) = inf[γ∈Γ(P,Q)] ∫ ||x - y||₁ dγ(x,y)
```

Où Γ(P,Q) est l'ensemble des couplages entre les distributions P et Q.

### 7.2 Indice de Modularité Adaptée

**Modularité des Similarités (MS)** :
```
MS = (1/2m) × Σᵢⱼ [Aᵢⱼ - (kᵢkⱼ/2m)] × δ(cᵢ, cⱼ)
```

Où :
- Aᵢⱼ = similarité entre embeddings i et j
- kᵢ = degré du nœud i
- δ(cᵢ, cⱼ) = 1 si i et j appartiennent au même cluster

## 8. Implémentation Pratique et Recommandations

### 8.1 Pipeline d'Évaluation Complet

```python
class EmbeddingSimilarityEvaluator:
    def __init__(self, ref_embeddings, doc_embeddings):
        self.ref_embeddings = ref_embeddings
        self.doc_embeddings = doc_embeddings
    
    def comprehensive_evaluation(self):
        results = {}
        
        # 1. Cohérence interne
        results['coherence'] = self.compute_coherence()
        
        # 2. Stabilité
        results['stability'] = self.compute_stability()
        
        # 3. Robustesse
        results['robustness'] = self.compute_robustness()
        
        # 4. Bootstrap
        results['confidence_intervals'] = self.bootstrap_analysis()
        
        # 5. Score composite
        results['composite_score'] = self.compute_composite_score(results)
        
        return results
    
    def compute_composite_score(self, individual_scores):
        weights = {'coherence': 0.3, 'stability': 0.25, 'robustness': 0.25, 'confidence': 0.2}
        
        normalized_scores = {}
        for metric, score in individual_scores.items():
            if metric != 'confidence_intervals':
                normalized_scores[metric] = self.normalize_score(score)
        
        composite = sum(weights[metric] * score for metric, score in normalized_scores.items())
        return composite
```

### 8.2 Seuils de Qualité Recommandés

| Métrique | Excellent | Bon | Acceptable | Médiocre |
|----------|-----------|-----|------------|----------|
| Cohérence (ICS) | < 0.1 | 0.1-0.3 | 0.3-0.5 | > 0.5 |
| Distance Intra-Cluster Normalisée | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |
| Distance Inter-Cluster Normalisée | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| Score Silhouette | > 0.7 | 0.5-0.7 | 0.3-0.5 | < 0.3 |
| Robustesse | > 0.9 | 0.8-0.9 | 0.7-0.8 | < 0.7 |
| Stabilité Bootstrap | σ < 0.05 | 0.05-0.1 | 0.1-0.2 | > 0.2 |

## 9. Cas Particuliers et Limitations

### 9.1 Embeddings de Dimensions Différentes
Si dim(E_ref) ≠ dim(E_doc), appliquer une transformation linéaire :
```
T: ℝᵈ¹ → ℝᵈ² minimisant ||E_ref × T - E_doc_projected||²
```

### 9.2 Gestion des Valeurs Aberrantes
Utiliser des métriques robustes comme la médiane absolue de déviation (MAD) :
```
MAD = médiane(|xᵢ - médiane(X)|)
```

## 10. Conclusion et Recommandations

L'évaluation de la qualité des similarités d'embeddings sans vérité terrain requiert une approche multi-métrique combinant :

1. **Cohérence interne** pour mesurer la consistance
2. **Stabilité statistique** pour évaluer la fiabilité
3. **Robustesse aux perturbations** pour tester la sensibilité
4. **Analyse comparative** pour contextualiser les résultats

**Recommandation pratique** : Utilisez un score composite pondéré intégrant les métriques de silhouette décomposées, la cohérence interne, la robustesse et l'analyse de densité locale pour obtenir une évaluation robuste de la qualité de vos calculs de similarité.