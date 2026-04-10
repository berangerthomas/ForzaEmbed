Evaluation Metrics
==================

ForzaEmbed uses clustering-based metrics to evaluate embedding quality.

Main Metrics Module
-------------------

.. automodule:: src.metrics.evaluation_metrics
   :members:
   :undoc-members:
   :show-inheritance:

Silhouette Analysis
-------------------

.. automodule:: src.metrics.silhouette_decomposition
   :members:
   :undoc-members:
   :show-inheritance:

Metric Descriptions
-------------------

Silhouette Score
~~~~~~~~~~~~~~~~

The silhouette score measures how well-defined the clusters are. It ranges from -1 to 1:

* **1.0**: Perfect clustering - points are far from other clusters
* **0.0**: Overlapping clusters - points are on cluster boundaries
* **-1.0**: Poor clustering - points may be assigned to wrong clusters

Formula:

.. math::

    s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}

where:

* :math:`a(i)` = average distance to points in the same cluster (intra-cluster)
* :math:`b(i)` = average distance to points in nearest different cluster (inter-cluster)

Intra-Cluster Distance (Normalized)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures cohesion within clusters. Normalized to [0, 1]:

* Higher values indicate tighter clustering
* Formula: :math:`1 - \\frac{\\text{avg\\_intra\\_distance}}{\\text{max\\_distance}}`

Inter-Cluster Distance (Normalized)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures separation between clusters. Normalized to [0, 1]:

* Higher values indicate better separation
* Formula: :math:`\\frac{\\text{avg\\_inter\\_distance}}{\\text{max\\_distance}}`

Embedding Computation Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures the total time (in seconds) required to compute embeddings for both:

* Theme keywords
* Document chunks

This metric helps identify performance bottlenecks across different embedding models and configurations.

Interpretation Guide
--------------------

Good Configuration
~~~~~~~~~~~~~~~~~~

A good embedding configuration should have:

* **Silhouette score** > 0.5
* **Intra-cluster distance** > 0.7
* **Inter-cluster distance** > 0.6
* **Embedding time** as low as possible for your use case

Poor Configuration
~~~~~~~~~~~~~~~~~~

Warning signs of poor configuration:

* Silhouette score < 0.3
* Intra-cluster distance < 0.5
* Inter-cluster distance < 0.4
* Large variance in cluster sizes
