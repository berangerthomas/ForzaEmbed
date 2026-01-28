Grid Search Optimization
========================

ForzaEmbed implements a grid search optimization that reduces computation time by avoiding redundant calculations.

Problem Statement
-----------------

In a naive grid search approach, all chunking strategies would be tested with all combinations of chunk_size and chunk_overlap parameters. However, some chunking strategies completely ignore these parameters because they use linguistic sentence boundaries instead of fixed sizes.

Testing these parameter-insensitive strategies with different chunk sizes produces **identical results**, wasting valuable computation time and storage space.

Chunking Strategy Classification
---------------------------------

ForzaEmbed classifies chunking strategies into two categories:

Parameter-Sensitive Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies use chunk_size and chunk_overlap parameters:

**langchain**
    Uses ``RecursiveCharacterTextSplitter`` with precise size control
    
    * Respects exact chunk_size limits
    * Implements chunk_overlap for context preservation
    * Splits recursively on separators while maintaining size constraints

**semchunk**
    Semantic chunking with size limits
    
    * Uses chunk_size as maximum limit
    * Maintains semantic coherence within size constraints
    * Produces different results with different sizes

**raw**
    Character-based chunking with overlap
    
    * Pure character-based splitting at chunk_size
    * Implements sliding window with chunk_overlap
    * Produces exact-sized chunks

Parameter-Insensitive Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These strategies ignore chunk_size and chunk_overlap:

**nltk**
    Uses ``nltk.sent_tokenize()`` for sentence segmentation
    
    * Splits on sentence boundaries using linguistic rules
    * Ignores chunk_size and chunk_overlap completely
    * Results depend only on text content and language

**spacy**
    Uses spaCy's sentence segmentation (``doc.sents``)
    
    * Splits using machine learning-based sentence detection
    * Ignores chunk_size and chunk_overlap completely
    * Results depend on linguistic model, not size parameters

Optimization Strategy
---------------------

The smart grid search optimization works as follows:

1. **Strategy Classification**
   
   Separates chunking strategies into sensitive and insensitive groups

2. **Combination Generation**
   
   * For **parameter-sensitive** strategies: Generate all valid combinations of chunk_size × chunk_overlap
   * For **parameter-insensitive** strategies: Generate only **one** combination using dummy values (first chunk_size and chunk_overlap)

3. **Validation**
   
   Ensures chunk_size > chunk_overlap for all sensitive strategy combinations

Performance Impact
------------------

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

Consider a typical configuration:

.. code-block:: yaml

    grid_search_params:
      chunk_size: [10, 20, 50, 100, 250, 500, 1000]  # 7 values
      chunk_overlap: [0, 5, 10, 25, 50, 100, 200]     # 7 values
      chunking_strategy: ["langchain", "raw", "semchunk", "nltk", "spacy"]
      similarity_metrics: ["cosine", "euclidean", "manhattan", "dot_product", "chebyshev"]
      themes:
        horaires: ["opening hours"]
        schedule: ["schedule", "timetable"]
        closures: ["closed on"]
    
    models_to_test:
      - type: "fastembed"
        name: "BAAI/bge-small-en-v1.5"
        dimensions: 384
      # ... 5 more models

Results
~~~~~~~

**Naive Approach (without optimization)**:

* Valid chunk pairs: 35 (from 7 × 7, filtered for size > overlap)
* Total combinations: 6 models × 35 pairs × 5 strategies × 5 metrics × 3 themes
* **Result: 15,750 combinations**

**Smart Approach (with optimization)**:

* Parameter-sensitive (langchain, raw, semchunk):
  
  * 6 models × 35 pairs × 3 strategies × 5 metrics × 3 themes = 9,450

* Parameter-insensitive (nltk, spacy):
  
  * 6 models × **1 pair** × 2 strategies × 5 metrics × 3 themes = 180

* **Total: 9,630 combinations**

**Optimization Results**:

* Combinations eliminated: **6,120**
* Reduction: **38.9%**
* Speedup: **1.64x**

Per-Strategy Savings
~~~~~~~~~~~~~~~~~~~~

For each parameter-insensitive strategy:

* Without optimization: 35 chunk configurations tested
* With optimization: 1 chunk configuration tested
* Savings: 34 configurations per strategy

With 6 models, 5 metrics, and 3 themes:

* Savings per insensitive strategy: (35 - 1) × 6 × 5 × 3 = **3,060 combinations**

Implementation Details
----------------------

The optimization is implemented in ``src/core/core.py``:

.. code-block:: python

    # Define parameter-insensitive strategies
    PARAMETER_INSENSITIVE_STRATEGIES = {"nltk", "spacy"}
    
    def _generate_smart_combinations(self, param_grid: dict) -> list:
        """
        Generates parameter combinations intelligently by avoiding
        redundant combinations for chunking strategies that don't
        use chunk_size/chunk_overlap.
        """
        # Separate strategies
        sensitive_strategies = [
            s for s in strategies 
            if s not in PARAMETER_INSENSITIVE_STRATEGIES
        ]
        insensitive_strategies = [
            s for s in strategies 
            if s in PARAMETER_INSENSITIVE_STRATEGIES
        ]
        
        # Generate all combinations for sensitive strategies
        # Generate only one combination for insensitive strategies
        # ...

Automatic Detection
-------------------

The optimization is **completely automatic**. No changes to your configuration files or API calls are required.

When running a grid search, you'll see logging output like:

.. code-block:: text

    Smart combination generation: 
      3 parameter-sensitive strategies (langchain, raw, semchunk), 
      2 parameter-insensitive strategies (nltk, spacy)

Transparency and Verification
------------------------------

To verify the optimization and understand the savings for your specific configuration, run:

.. code-block:: bash

    python demo_smart_optimization.py

This script displays:

* Strategy classification
* Combination counts (naive vs optimized)
* Reduction percentage and speedup factor
* Detailed per-strategy breakdown

Best Practices
--------------

Maximize Optimization Benefits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To maximize the benefits of this optimization:

1. **Include both types of strategies** in your configuration
   
   Mix parameter-sensitive (langchain, semchunk, raw) with parameter-insensitive (nltk, spacy)

2. **Use multiple chunk_size and chunk_overlap values**
   
   The more values you test, the greater the savings for insensitive strategies

3. **Test many configurations**
   
   With multiple models, metrics, and themes, the multiplicative effect increases savings

When to Use Each Strategy Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use parameter-sensitive strategies** when:

* You need precise control over chunk size
* Working with structured data or code
* Chunk size significantly impacts your use case
* You want to test different granularities

**Use parameter-insensitive strategies** when:

* Natural sentence boundaries are important
* Working with narrative or conversational text
* Linguistic coherence is a priority
* You want grammatically complete chunks

Technical References
--------------------

Sentence Tokenization
~~~~~~~~~~~~~~~~~~~~~

* **NLTK**: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize
* **spaCy**: https://spacy.io/usage/linguistic-features#sbd

Character-Based Chunking
~~~~~~~~~~~~~~~~~~~~~~~~~

* **LangChain**: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
* **SemChunk**: https://github.com/umarbutler/semchunk
