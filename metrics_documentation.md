# Métriques d'évaluation pour le filtrage de textes

## 1. Métriques de clustering spécifiques

### Cohésion (Cohesion)

*   **Définition** : La cohésion mesure la similarité sémantique moyenne à l'intérieur d'un même thème. Elle quantifie à quel point les textes assignés à un thème sont proches les uns des autres.
*   **Calcul** : Pour chaque thème, nous calculons la similarité cosinus moyenne entre toutes les paires de textes uniques. La cohésion globale est la moyenne de ces scores sur l'ensemble des thèmes.
*   **Interprétation** : Un score de **cohésion élevé** (proche de 1) est souhaitable. Il indique que les textes regroupés sous un même thème sont sémantiquement homogènes et traitent du même sujet.
*   **Application Concrète** : Une cohésion forte valide la pertinence d'un thème. Si un thème présente une faible cohésion, cela peut signifier qu'il est trop large, mal défini, ou qu'il contient des textes hors-sujet qui devraient être filtrés ou réassignés.

### Séparation (Separation)

*   **Définition** : La séparation mesure la dissimilarité sémantique moyenne entre les différents thèmes. Elle évalue à quel point les thèmes sont distincts les uns des autres.
*   **Calcul** : Nous calculons la similarité cosinus moyenne entre les textes de chaque paire de thèmes distincts. La séparation globale est la moyenne de ces scores.
*   **Interprétation** : Un score de **séparation bas** (proche de 0) est souhaitable. Il indique que les thèmes sont bien différenciés et ne se chevauchent pas sémantiquement.
*   **Application Concrète** : Une faible séparation garantit que notre système de filtrage peut distinguer clairement les sujets. Si deux thèmes ont une similarité élevée, ils pourraient être fusionnés ou leurs définitions devraient être affinées pour éviter les ambiguïtés.

### Score Discriminant (Discriminant Score)

*   **Définition** : Le score discriminant est un ratio qui combine la cohésion et la séparation en une seule métrique globale.
*   **Calcul** : Il est calculé comme le rapport `Cohésion / Séparation`.
*   **Interprétation** : Un **score discriminant élevé** est souhaitable. Il indique un équilibre optimal : des thèmes à la fois denses (forte cohésion) et bien distincts les uns des autres (faible séparation).
*   **Application Concrète** : Cette métrique est l'indicateur principal pour comparer la performance globale de différents modèles d'embedding ou de différentes stratégies de clustering. Un modèle qui maximise ce score est celui qui structure le mieux l'information pour un filtrage thématique efficace.

---

## 2. Métriques de Clustering Standards

Ces métriques sont des standards de l'industrie pour l'évaluation de modèles de clustering non supervisés. Elles fournissent une perspective complémentaire sur la structure des données.

### Score de Silhouette (Silhouette Score)

*   **Définition** : Le score de Silhouette mesure pour chaque texte à quel point il est bien intégré à son propre thème par rapport aux autres thèmes. Il prend en compte à la fois la distance intra-thème et la distance inter-thèmes.
*   **Interprétation** : Le score varie de -1 à +1.
    *   **Proche de +1** : Le texte est très bien assigné à son thème et loin des autres. C'est le cas idéal.
    *   **Proche de 0** : Le texte est à la frontière entre deux thèmes.
    *   **Proche de -1** : Le texte est probablement mal classé et est plus proche d'un autre thème.
*   **Application Concrète** : Le score de Silhouette est excellent pour une analyse fine au niveau du texte. Il permet d'identifier les documents "ambigus" ou mal classifiés qui pourraient nécessiter une vérification manuelle ou un retraitement. Un score moyen élevé pour l'ensemble des données indique un clustering de bonne qualité.

### Score de Calinski-Harabasz (Calinski-Harabasz Score)

*   **Définition** : Aussi connu sous le nom de "Variance Ratio Criterion", ce score évalue la qualité du clustering en comparant la dispersion entre les thèmes (variance inter-cluster) à la dispersion à l'intérieur des thèmes (variance intra-cluster).
*   **Interprétation** : Un **score plus élevé est meilleur**. Il signifie que les thèmes sont denses et bien séparés les uns des autres.
*   **Application Concrète** : C'est une métrique rapide et efficace pour juger de la qualité globale d'un partitionnement. Elle est particulièrement utile pour comparer des modèles : un modèle d'embedding qui produit un score de Calinski-Harabasz plus élevé est généralement meilleur pour séparer les concepts.

### Score de Davies-Bouldin (Davies-Bouldin Score)

*   **Définition** : Ce score mesure la "similarité" moyenne de chaque thème avec son thème le plus proche. La similarité est définie comme le rapport de la somme des distances intra-thème sur la distance inter-thèmes.
*   **Interprétation** : Un **score plus bas est meilleur**, avec 0 comme score parfait. Un score faible indique que les thèmes sont bien séparés, même de leurs voisins les plus proches.
*   **Application Concrète** : Similaire au score de Calinski-Harabasz, il évalue la séparation des clusters. Il est particulièrement sensible aux clusters qui se touchent. Un faible score de Davies-Bouldin nous donne confiance dans le fait que même les thèmes les plus similaires dans notre corpus sont suffisamment distincts pour être traités séparément.
