# Evaluation Metrics for Text Filtering

## 1. Specific Clustering Metrics

### Cohesion

*   **Definition**: Cohesion measures the average semantic similarity within a single topic. It quantifies how close the texts assigned to a topic are to each other.
*   **Computation**: For each topic, we compute the average cosine similarity between all unique pairs of texts. The overall cohesion is the mean of these scores across all topics.
*   **Interpretation**: A **high cohesion score** (close to 1) is desirable. It indicates that the texts grouped under the same topic are semantically homogeneous and address the same subject.
*   **Practical Application**: Strong cohesion validates the relevance of a topic. If a topic shows low cohesion, it may mean that it is too broad, poorly defined, or contains off-topic texts that should be filtered or reassigned.

### Separation

*   **Definition**: Separation measures the average semantic dissimilarity between different topics. It evaluates how distinct the topics are from each other.
*   **Computation**: We compute the average cosine similarity between texts from each pair of distinct topics. The overall separation is the mean of these scores.
*   **Interpretation**: A **low separation score** (close to 0) is desirable. It indicates that the topics are well differentiated and do not overlap semantically.
*   **Practical Application**: Low separation ensures that our filtering system can clearly distinguish between subjects. If two topics have high similarity, they could be merged or their definitions should be refined to avoid ambiguities.

### Discriminant Score

*   **Definition**: The discriminant score is a ratio that combines cohesion and separation into a single global metric.
*   **Computation**: It is calculated as the ratio `Cohesion / Separation`.
*   **Interpretation**: A **high discriminant score** is desirable. It indicates an optimal balance: topics that are both dense (high cohesion) and well separated from each other (low separation).
*   **Practical Application**: This metric is the main indicator for comparing the overall performance of different embedding models or clustering strategies. A model that maximizes this score is the one that best structures the information for effective thematic filtering.

---

## 2. Standard Clustering Metrics

These metrics are industry standards for evaluating unsupervised clustering models. They provide a complementary perspective on the data structure.

### Silhouette Score

*   **Definition**: The Silhouette score measures for each text how well it fits into its own topic compared to other topics. It takes into account both intra-topic and inter-topic distances.
*   **Interpretation**: The score ranges from -1 to +1.
    *   **Close to +1**: The text is very well assigned to its topic and far from others. This is the ideal case.
    *   **Close to 0**: The text is at the boundary between two topics.
    *   **Close to -1**: The text is probably misclassified and is closer to another topic.
*   **Practical Application**: The Silhouette score is excellent for fine-grained analysis at the text level. It helps identify "ambiguous" or misclassified documents that may require manual review or reprocessing. A high average score for the entire dataset indicates good clustering quality.

### Calinski-Harabasz Score

*   **Definition**: Also known as the "Variance Ratio Criterion", this score evaluates the quality of clustering by comparing the dispersion between topics (inter-cluster variance) to the dispersion within topics (intra-cluster variance).
*   **Interpretation**: A **higher score is better**. It means that the topics are dense and well separated from each other.
*   **Practical Application**: This is a fast and effective metric for judging the overall quality of a partitioning. It is particularly useful for comparing models: an embedding model that produces a higher Calinski-Harabasz score is generally better at separating concepts.

### Davies-Bouldin Score

*   **Definition**: This score measures the average "similarity" of each topic with its closest topic. Similarity is defined as the ratio of the sum of intra-topic distances to the inter-topic distance.
*   **Interpretation**: A **lower score is better**, with 0 as the perfect score. A low score indicates that the topics are well separated, even from their closest neighbors.
*   **Practical Application**: Similar to the Calinski-Harabasz score, it evaluates cluster separation. It is particularly sensitive to clusters that touch each other. A low Davies-Bouldin score gives us confidence that even the most similar topics in our corpus are sufficiently distinct to be treated separately.

