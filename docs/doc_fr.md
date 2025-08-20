# Métriques d'Évaluation du Framework

## Introduction

Ce document fournit une description technique des métriques d'évaluation utilisées au sein du framework ForzaEmbed. Ces métriques sont conçues pour évaluer la qualité des embeddings de texte et des structures de clustering qui en résultent, de manière non supervisée, c'est-à-dire sans nécessiter de données de référence pré-étiquetées.

---

## 1. Métriques de Qualité du Clustering

Ces métriques évaluent les propriétés géométriques et la cohérence des clusters formés par les embeddings de documents.

### 1.1. Score Silhouette

*   **Définition** : Le score Silhouette quantifie la pertinence de l'assignation d'un point de donnée à son cluster par rapport aux autres clusters. Il fournit une mesure à la fois de la cohésion intra-cluster (similarité des points au sein d'un cluster) et de la séparation inter-cluster (distinction entre les différents clusters).

*   **Formule** : Pour un unique point de donnée `i` :
    $$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
    Où :
    -   `a(i)` est la distance moyenne de `i` à tous les autres points du même cluster (distance intra-cluster).
    -   `b(i)` est la distance moyenne de `i` à tous les points du cluster voisin le *plus proche* (distance inter-cluster).
    Le score global est la moyenne de `s(i)` sur l'ensemble des points.

*   **Interprétation** : Le score varie de -1 à +1.
    *   **+1** : Indique des clusters denses et bien séparés.
    *   **0** : Indique des clusters qui se chevauchent ou des points situés à la frontière.
    *   **-1** : Indique que les points ont probablement été assignés au mauvais cluster.

*   **Cas d'Usage** : Fournit une évaluation globale et de haut niveau de la qualité de la structure de clustering pour un modèle d'embedding et une stratégie de segmentation donnés.

### 1.2. Composantes du Score Silhouette Décomposées

Pour offrir un diagnostic plus granulaire, les composantes principales du score Silhouette sont analysées indépendamment.

#### Qualité Intra-Cluster (Cohésion)

*   **Définition** : Mesure la similarité sémantique moyenne au sein des clusters. Une valeur élevée indique que les documents assignés à un même thème sont sémantiquement homogènes.

*   **Formule** : Cette métrique est une version normalisée de la distance intra-cluster moyenne `a(i)`.
    $$ \text{Cohésion} = 1 - \frac{\text{moyenne}(a(i))}{\text{dist_max}} $$
    Où `dist_max` est un facteur de normalisation représentant la distance maximale possible dans l'espace.

*   **Interprétation** : Un score **proche de 1 est meilleur**, signifiant une cohésion interne élevée au sein des clusters.

#### Séparation Inter-Cluster

*   **Définition** : Mesure la dissimilarité sémantique moyenne entre un cluster et son plus proche voisin. Elle évalue à quel point les thèmes sont bien définis et distincts les uns des autres.

*   **Formule** : Cette métrique est une version normalisée de la distance moyenne au cluster le plus proche `b(i)`.
    $$ \text{Séparation} = \frac{\text{moyenne}(b(i))}{\text{dist_max}} $$

*   **Interprétation** : Un score **proche de 1 est meilleur**, indiquant que les clusters sont bien séparés dans l'espace des embeddings.

---

## 2. Métriques de Stabilité de l'Espace d'Embedding et du Système

Ces métriques évaluent les propriétés intrinsèques de l'espace des embeddings et la stabilité du système de mesure de similarité.

### 2.1. Score de Cohérence Interne (ICS)

*   **Définition** : Évalue la stabilité et la prévisibilité du système de similarité. Un système cohérent doit produire des scores de similarité constants pour un thème donné à travers tous les segments de document, plutôt que des scores très variables.

*   **Formule** : C'est le ratio moyen de la variance sur la moyenne des scores de similarité pour chaque thème de référence `t` par rapport à tous les segments de document `D`.
    $$ \text{ICS} = \frac{1}{|T|} \sum_{t \in T} \frac{\text{Var}(\text{sim}(t, d_i) \text{ pour } d_i \in D)}{\text{Moyenne}(\text{sim}(t, d_i) \text{ pour } d_i \in D)} $$

*   **Interprétation** : Un **score plus faible est meilleur**. Un score bas indique que la mesure de similarité est stable et ne produit pas de résultats erratiques.

### 2.2. Indice de Densité Locale (LDI)

*   **Définition** : Évalue la structure locale de l'espace des embeddings en mesurant, pour chaque point, la proportion de ses plus proches voisins qui appartiennent au même cluster.

*   **Formule** : Pour un ensemble de points `X` avec des labels `L`, et pour chaque point `x_i`, soit `N_k(x_i)` l'ensemble de ses `k` plus proches voisins.
    $$ \text{LDI} = \frac{1}{|X|} \sum_{i=1}^{|X|} \frac{|\{x_j \in N_k(x_i) \mid L(x_j) = L(x_i)\}|}{k} $$

*   **Interprétation** : Un **score plus élevé est meilleur**. Un score de 1.0 indique que pour chaque point, tous ses `k` plus proches voisins partagent le même thème, suggérant un espace d'embedding bien structuré.

### 2.3. Score de Robustesse (RS)

*   **Définition** : Teste la stabilité du résultat du clustering en introduisant une petite quantité de bruit gaussien aléatoire aux embeddings et en mesurant le changement dans le score Silhouette qui en résulte.

*   **Formule** :
    $$ \text{RS} = 1 - \frac{|\text{score}_{\text{original}} - \text{score}_{\text{perturbé}}|}{\max(|\text{score}_{\text{original}}|, \epsilon)} $$
    Où `score_perturbé` est le score moyen sur plusieurs injections de bruit et `epsilon` est une petite constante pour éviter la division par zéro.

*   **Interprétation** : Un **score proche de 1.0 est meilleur**, indiquant que le clustering est stable et peu sensible aux perturbations mineures dans l'espace des embeddings.