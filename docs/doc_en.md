# Framework Evaluation Metrics

## Introduction

This document provides a technical description of the evaluation metrics used within the ForzaEmbed framework. These metrics are designed to assess the quality of text embeddings and the resulting clustering structures in an unsupervised manner, meaning they do not require pre-labeled ground-truth data.

---

## 1. Clustering Quality Metrics

These metrics evaluate the geometric properties and coherence of the clusters formed by the document embeddings.

### 1.1. Silhouette Score

*   **Definition**: The Silhouette Score quantifies how well a data point is assigned to its cluster relative to other clusters. It provides a measure of both cluster cohesion (how similar points are within a cluster) and separation (how distinct different clusters are).

*   **Formula**: For a single data point `i`:
    $$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
    Where:
    -   `a(i)` is the mean distance from `i` to all other points in the same cluster (intra-cluster distance).
    -   `b(i)` is the mean distance from `i` to all points in the *nearest* neighboring cluster (inter-cluster distance).
    The overall score is the average `s(i)` over all data points.

*   **Interpretation**: The score ranges from -1 to +1.
    *   **+1**: Indicates dense, well-separated clusters.
    *   **0**: Indicates overlapping clusters or points on the boundary.
    *   **-1**: Indicates that points may have been assigned to the wrong cluster.

*   **Use Case**: Provides a global, high-level assessment of the clustering structure's quality for a given embedding model and chunking strategy.

### 1.2. Decomposed Silhouette Components

To offer a more granular diagnosis, the core components of the Silhouette Score are analyzed independently.

#### Intra-Cluster Quality (Cohesion)

*   **Definition**: Measures the average semantic similarity within clusters. A high value indicates that documents assigned to the same theme are semantically homogeneous.

*   **Formula**: This metric is a normalized version of the average intra-cluster distance `a(i)`.
    $$ \text{Cohesion} = 1 - \frac{\text{mean}(a(i))}{\text{max_dist}} $$
    Where `max_dist` is a normalization factor representing the maximum possible distance in the space.

*   **Interpretation**: A score **closer to 1 is better**, signifying high internal cohesion within clusters.

#### Inter-Cluster Separation

*   **Definition**: Measures the average semantic dissimilarity between a cluster and its nearest neighbor. It evaluates how well-defined and distinct the themes are from one another.

*   **Formula**: This metric is a normalized version of the average nearest-cluster distance `b(i)`.
    $$ \text{Separation} = \frac{\text{mean}(b(i))}{\text{max_dist}} $$

*   **Interpretation**: A score **closer to 1 is better**, indicating that clusters are well-separated in the embedding space.

---

## 2. Embedding Space & System Stability Metrics

These metrics evaluate the intrinsic properties of the embedding space and the stability of the similarity measurement system.

### 2.1. Internal Coherence Score (ICS)

*   **Definition**: Assesses the stability and predictability of the similarity system. A coherent system should produce consistent similarity scores for a given theme across all document chunks, rather than highly variable scores.

*   **Formula**: It is the average ratio of variance to the mean of similarity scores for each reference theme `t` against all document chunks `D`.
    $$ \text{ICS} = \frac{1}{|T|} \sum_{t \in T} \frac{\text{Var}(\text{sim}(t, d_i) \text{ for } d_i \in D)}{\text{Mean}(\text{sim}(t, d_i) \text{ for } d_i \in D)} $$

*   **Interpretation**: A **lower score is better**. A low score indicates that the similarity measure is stable and not producing erratic results.

### 2.2. Local Density Index (LDI)

*   **Definition**: Evaluates the local structure of the embedding space by measuring, for each point, the proportion of its nearest neighbors that belong to the same cluster.

*   **Formula**: For a set of points `X` with labels `L`, and for each point `x_i`, let `N_k(x_i)` be the set of its `k` nearest neighbors.
    $$ \text{LDI} = \frac{1}{|X|} \sum_{i=1}^{|X|} \frac{|\{x_j \in N_k(x_i) \mid L(x_j) = L(x_i)\}|}{k} $$

*   **Interpretation**: A **higher score is better**. A score of 1.0 indicates that for every point, all of its `k` nearest neighbors share the same theme, suggesting a well-structured embedding space.

### 2.3. Robustness Score (RS)

*   **Definition**: Tests the stability of the clustering outcome by introducing a small amount of random Gaussian noise to the embeddings and measuring the change in the resulting Silhouette Score.

*   **Formula**:
    $$ \text{RS} = 1 - \frac{|\text{score}_{\text{original}} - \text{score}_{\text{perturbed}}|}{\max(|\text{score}_{\text{original}}|, \epsilon)} $$
    Where `score_perturbed` is the average score over multiple noise injections and `epsilon` is a small constant to prevent division by zero.

*   **Interpretation**: A **score closer to 1.0 is better**, indicating that the clustering is stable and not highly sensitive to minor perturbations in the embedding space.