// ForzaEmbed — Main interactive report logic
// Data is injected by the Python generator as: b64DataChunks, themesConfig, workerScript

'use strict';

// ---------------------------------------------------------------------------
// Tooltip definitions
// ---------------------------------------------------------------------------
const metricTooltips = {
    'intra_cluster_distance_normalized': "Intra-Cluster Cohesion. Measures how tightly grouped texts of the same theme are. HIGHER IS BETTER. (> 0.8: Excellent, 0.6-0.8: Good, 0.4-0.6: Fair, < 0.4: Poor)",
    'inter_cluster_distance_normalized': "Inter-Cluster Separation. Measures how well different themes are separated. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, < 0.3: Poor)",
    'silhouette_score': "Silhouette Score. Global clustering quality combining cohesion and separation. Range: -1 to 1. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, 0-0.3: Poor, < 0: Very Poor)",
    'embedding_computation_time': "Embedding Computation Time (seconds). Time required to compute embeddings for both themes and chunks. LOWER IS BETTER."
};

const sliderTooltips = {
    'file': "Select the markdown file to analyse. Each file is processed independently with all parameter combinations.",
    'model': "Embedding model used to convert text into vectors. Different models have varying quality, speed, and dimensionality.",
    'chunk-size': "Size of text segments in characters. Smaller chunks = finer analysis but more noise. Larger chunks = more context but less precision.",
    'chunk-overlap': "Overlap between consecutive chunks in characters. Helps preserve context across chunk boundaries.",
    'theme': "Set of keywords defining what semantic content to search for. Chunks are scored by similarity to these themes.",
    'chunking-strategy': "Algorithm used to split text into chunks. Different strategies handle sentence boundaries and semantic units differently.",
    'similarity-metric': "Mathematical method to measure distance/similarity between embedding vectors."
};

const valueTooltips = {
    'chunking-strategy': {
        'raw': "Simple character-based splitting. Fast but may cut words and sentences arbitrarily.",
        'langchain': "Recursive character splitter from LangChain. Tries to split on paragraphs, then sentences, then words.",
        'semchunk': "Semantic chunking that respects sentence boundaries and tries to keep related content together.",
        'nltk': "Uses NLTK sentence tokenizer. Each sentence becomes a chunk (ignores chunk_size/overlap).",
        'spacy': "Uses spaCy NLP pipeline for advanced sentence segmentation (ignores chunk_size/overlap)."
    },
    'similarity-metric': {
        'cosine': "Cosine similarity. Measures angle between vectors (0-1). Most common, normalised for vector magnitude.",
        'dot_product': "Dot product. Combines angle and magnitude. Good for models trained with dot product loss.",
        'euclidean': "Euclidean distance (L2). Straight-line distance in vector space. Converted to similarity.",
        'manhattan': "Manhattan distance (L1). Sum of absolute differences. More robust to outliers.",
        'chebyshev': "Chebyshev distance (L∞). Maximum difference along any dimension."
    }
};

// ---------------------------------------------------------------------------
// DOM element references
// ---------------------------------------------------------------------------
const fileSlider              = document.getElementById('file-slider');
const fileNameSpan            = document.getElementById('file-name');
const modelSlider             = document.getElementById('model-slider');
const modelNameSpan           = document.getElementById('model-name');
const chunkSizeSlider         = document.getElementById('chunk-size-slider');
const chunkSizeValueSpan      = document.getElementById('chunk-size-value');
const chunkOverlapSlider      = document.getElementById('chunk-overlap-slider');
const chunkOverlapValueSpan   = document.getElementById('chunk-overlap-value');
const themeSlider             = document.getElementById('theme-slider');
const themeNameSpan           = document.getElementById('theme-name');
const chunkingStrategySlider  = document.getElementById('chunking-strategy-slider');
const chunkingStrategyNameSpan= document.getElementById('chunking-strategy-name');
const similarityMetricSlider  = document.getElementById('similarity-metric-slider');
const similarityMetricNameSpan= document.getElementById('similarity-metric-name');
const metricsGrid             = document.getElementById('metrics-grid');
const heatmapContainer        = document.getElementById('heatmap-container');
const scatterPlotContainer    = document.getElementById('scatter-plot-container');
const fileLinksContainer      = document.getElementById('file-links-container');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let processedData = {};
let fileKeys = [];
let allEmbeddingKeys = [];
let filteredEmbeddingKeys = [];
let activeProjection = "tsne";
const params = { model: [], cs: [], co: [], t: [], s: [], m: [] };

// ---------------------------------------------------------------------------
// Color utilities — professional, muted palette
// ---------------------------------------------------------------------------
const createCmap = (colorSet) => (score) => {
    if (typeof score !== 'number' || isNaN(score)) score = 0.0;
    score = Math.max(0.0, Math.min(1.0, score));

    if (score === 1.0) {
        const c = colorSet[colorSet.length - 1];
        return { rgb: `rgb(${c.r},${c.g},${c.b})`, isDark: (c.r * 0.299 + c.g * 0.587 + c.b * 0.114) < 128 };
    }

    const scaledScore = score * (colorSet.length - 1);
    const i = Math.floor(scaledScore);
    const t = scaledScore - i;
    const c1 = colorSet[i];
    const c2 = colorSet[i + 1];

    const r = Math.round(c1.r * (1 - t) + c2.r * t);
    const g = Math.round(c1.g * (1 - t) + c2.g * t);
    const b = Math.round(c1.b * (1 - t) + c2.b * t);

    return { rgb: `rgb(${r},${g},${b})`, isDark: (r * 0.299 + g * 0.587 + b * 0.114) < 128 };
};

// Heatmap: RdYlBu-inspired — clear gradient, not too pastel, not too neon
const cmap_heatmap = createCmap([
    { r: 58,  g: 112, b: 189 },  // Medium blue    — low similarity
    { r: 108, g: 187, b: 170 },  // Soft teal      — below average
    { r: 254, g: 246, b: 140 },  // Muted gold     — neutral
    { r: 240, g: 140, b: 50  },  // Warm amber     — above average
    { r: 200, g: 40,  b: 40  }   // Deep red       — high similarity
]);

function getMetricColor(metricKey, value) {
    // Muted: dark-slate red → pale gold → forest green
    const cmap_metrics = createCmap([
        { r: 192, g: 62,  b: 62  },  // Muted red
        { r: 250, g: 235, b: 170 },  // Pale gold
        { r: 46,  g: 130, b: 86  }   // Forest green
    ]);

    const metricConfigs = {
        'intra_cluster_distance_normalized':  { min: 0.4, max: 1.0, lowerIsBetter: false },
        'inter_cluster_distance_normalized':  { min: 0.3, max: 0.8, lowerIsBetter: false },
        'silhouette_score':                   { min: -0.1, max: 0.7, lowerIsBetter: false },
        'embedding_computation_time':         { min: 0.0, max: 10.0, lowerIsBetter: true  }
    };

    const config = metricConfigs[metricKey];
    if (!config) return { rgb: '#f1f3f5', isDark: false };

    const clampedValue = Math.max(config.min, Math.min(config.max, value));
    let normalizedScore;
    if (config.max === config.min) {
        normalizedScore = 0.5;
    } else if (config.lowerIsBetter) {
        normalizedScore = (config.max - clampedValue) / (config.max - config.min);
    } else {
        normalizedScore = (clampedValue - config.min) / (config.max - config.min);
    }

    const result = cmap_metrics(normalizedScore);
    result.normalizedScore = normalizedScore;
    return result;
}

// ---- t-SNE Metadata coloring ----
function getTSNEColor(metricKey, value) {
    if (value === 'N/A' || value == null) return { rgb: '#f1f5f9' };
    
    // Muted: dark-slate red → pale gold → forest green
    const cmap_metrics = createCmap([
        { r: 192, g: 62,  b: 62  },  // Muted red
        { r: 250, g: 235, b: 170 },  // Pale gold
        { r: 46,  g: 130, b: 86  }   // Forest green
    ]);

    // For KL Divergence, lower is better. Range typical: 0.0 (perfect) to ~2.0+ (poor)
    if (metricKey === 'kl_divergence') {
        const min = 0.0;
        const max = 2.0; 
        const clampedValue = Math.max(min, Math.min(max, value));
        const normalizedScore = (max - clampedValue) / (max - min); // Inverse: lower value = higher score
        return cmap_metrics(normalizedScore);
    }
    
    return { rgb: '#f1f5f9' }; // default for Iterations and Perplexity
}

// ---------------------------------------------------------------------------
// Key parsing
// ---------------------------------------------------------------------------
function parseEmbeddingKey(key) {
    const m_part_index = key.lastIndexOf('_m');
    const m = key.slice(m_part_index + 2);
    const key_without_m = key.slice(0, m_part_index);

    const s_part_index = key_without_m.lastIndexOf('_s');
    const s = key_without_m.slice(s_part_index + 2);
    const key_without_s = key_without_m.slice(0, s_part_index);

    const t_part_index = key_without_s.lastIndexOf('_t');
    const t = key_without_s.slice(t_part_index + 2);
    const key_without_t = key_without_s.slice(0, t_part_index);

    const co_part_index = key_without_t.lastIndexOf('_co');
    const co = key_without_t.slice(co_part_index + 3);
    const key_without_co = key_without_t.slice(0, co_part_index);

    const cs_part_index = key_without_co.lastIndexOf('_cs');
    const cs = key_without_co.slice(cs_part_index + 3);
    const model = key_without_co.slice(0, cs_part_index);

    return { model, cs, co, t, s, m };
}

// ---------------------------------------------------------------------------
// Slider setup helpers
// ---------------------------------------------------------------------------
function populateAndSetupSliders(fileKey) {
    if (!fileKey || !processedData.files[fileKey]) return;

    const currentValues = {
        model: params.model[modelSlider.value],
        cs: params.cs[chunkSizeSlider.value],
        co: params.co[chunkOverlapSlider.value],
        t:  params.t[themeSlider.value],
        s:  params.s[chunkingStrategySlider.value],
        m:  params.m[similarityMetricSlider.value]
    };

    const paramSets = { model: new Set(), cs: new Set(), co: new Set(), t: new Set(), s: new Set(), m: new Set() };

    allEmbeddingKeys = Object.keys(processedData.files[fileKey].embeddings);
    allEmbeddingKeys.forEach(key => {
        const p = parseEmbeddingKey(key);
        paramSets.model.add(p.model);
        paramSets.cs.add(p.cs);
        paramSets.co.add(p.co);
        paramSets.t.add(p.t);
        paramSets.s.add(p.s);
        paramSets.m.add(p.m);
    });

    params.model = [...paramSets.model].sort();
    params.cs    = [...paramSets.cs].sort((a, b) => a - b);
    params.co    = [...paramSets.co].sort((a, b) => a - b);
    params.t     = [...paramSets.t];
    params.s     = [...paramSets.s];
    params.m     = [...paramSets.m];

    const setupSlider = (slider, values, previousValue) => {
        slider.max = values.length > 0 ? values.length - 1 : 0;
        slider.disabled = values.length <= 1;
        const newIndex = values.indexOf(previousValue);
        slider.value = newIndex !== -1 ? newIndex : 0;
    };

    setupSlider(modelSlider,            params.model, currentValues.model);
    setupSlider(chunkSizeSlider,        params.cs,    currentValues.cs);
    setupSlider(chunkOverlapSlider,     params.co,    currentValues.co);
    setupSlider(themeSlider,            params.t,     currentValues.t);
    setupSlider(chunkingStrategySlider, params.s,     currentValues.s);
    setupSlider(similarityMetricSlider, params.m,     currentValues.m);
}

function applySliderTooltips() {
    [
        { slider: fileSlider,              key: 'file'              },
        { slider: modelSlider,             key: 'model'             },
        { slider: chunkSizeSlider,         key: 'chunk-size'        },
        { slider: chunkOverlapSlider,      key: 'chunk-overlap'     },
        { slider: themeSlider,             key: 'theme'             },
        { slider: chunkingStrategySlider,  key: 'chunking-strategy' },
        { slider: similarityMetricSlider,  key: 'similarity-metric' }
    ].forEach(({ slider, key }) => {
        const group = slider.closest('.control-group');
        if (group && sliderTooltips[key]) group.title = sliderTooltips[key];
    });
}

function updateValueTooltip(sliderKey, value, spanElement) {
    spanElement.title = (valueTooltips[sliderKey] && valueTooltips[sliderKey][value])
        ? valueTooltips[sliderKey][value]
        : '';
}

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------
function initialize() {
    fileKeys = Object.keys(processedData.files || {}).sort();
    if (fileKeys.length === 0) {
        console.error('No files found in processed data.');
        return;
    }

    applySliderTooltips();

    fileSlider.max = fileKeys.length > 0 ? fileKeys.length - 1 : 0;
    fileSlider.disabled = fileKeys.length <= 1;
    fileSlider.value = 0;

    updateView(fileKeys[0], true);

    fileSlider.addEventListener('input', (e) => {
        updateView(fileKeys[parseInt(e.target.value, 10)], true);
    });

    [modelSlider, chunkSizeSlider, chunkOverlapSlider, themeSlider,
     chunkingStrategySlider, similarityMetricSlider].forEach(slider => {
        slider.addEventListener('input', () => {
            updateView(fileKeys[parseInt(fileSlider.value, 10)]);
        });
    });

    ['tsne', 'umap', 'pca'].forEach(proj => {
        const btn = document.getElementById('btn-' + proj);
        if (btn) {
            btn.addEventListener('click', () => {
                activeProjection = proj;
                document.querySelectorAll('.proj-btn').forEach(b => b.classList.remove('active', 'btn-selected'));
                document.querySelectorAll('.proj-btn').forEach(b => {b.style.background = 'white'; b.style.color = '#475569';});
                btn.style.background = '#3b82f6';
                btn.style.color = 'white';
                // Trigger re-render of currently selected data
                if (fileKeys.length > 0) {
                    updateView(fileKeys[parseInt(fileSlider.value, 10)]);
                }
            });
        }
    });



    [modelSlider, chunkSizeSlider, chunkOverlapSlider, themeSlider,
     chunkingStrategySlider, similarityMetricSlider].forEach(slider => {
        slider.addEventListener('input', () => {
            updateView(fileKeys[parseInt(fileSlider.value, 10)]);
        });
    });
}

// ---------------------------------------------------------------------------
// Filter embeddings based on current slider state
// ---------------------------------------------------------------------------
function filterEmbeddings() {
    const selectedModel = params.model[modelSlider.value];
    const selectedCS    = params.cs[chunkSizeSlider.value];
    const selectedCO    = params.co[chunkOverlapSlider.value];
    const selectedT     = params.t[themeSlider.value];
    const selectedS     = params.s[chunkingStrategySlider.value];
    const selectedM     = params.m[similarityMetricSlider.value];

    modelNameSpan.textContent           = selectedModel || 'N/A';
    chunkSizeValueSpan.textContent      = selectedCS    || 'N/A';
    chunkOverlapValueSpan.textContent   = selectedCO    || 'N/A';
    themeNameSpan.textContent           = selectedT     || 'N/A';
    chunkingStrategyNameSpan.textContent= selectedS     || 'N/A';
    similarityMetricNameSpan.textContent= selectedM     || 'N/A';

    updateValueTooltip('chunking-strategy', selectedS, chunkingStrategyNameSpan);
    updateValueTooltip('similarity-metric', selectedM, similarityMetricNameSpan);

    if (selectedT && themesConfig && themesConfig[selectedT]) {
        themeNameSpan.title = 'Keywords: ' + themesConfig[selectedT].join(', ');
    } else {
        themeNameSpan.title = '';
    }

    filteredEmbeddingKeys = allEmbeddingKeys.filter(key => {
        const p = parseEmbeddingKey(key);
        return p.model === selectedModel &&
               p.cs    === selectedCS    &&
               p.co    === selectedCO    &&
               p.t     === selectedT     &&
               p.s     === selectedS     &&
               p.m     === selectedM;
    });
}

// ---------------------------------------------------------------------------
// Find best parameter combination for a given metric
// ---------------------------------------------------------------------------
function findBestAndApply(metricKey) {
    const fileKey = fileKeys[parseInt(fileSlider.value, 10)];
    if (!fileKey || !processedData.files[fileKey]) return;

    const metricConfig = {
        'intra_cluster_distance_normalized': { lowerIsBetter: false },
        'inter_cluster_distance_normalized': { lowerIsBetter: false },
        'silhouette_score':                  { lowerIsBetter: false },
        'embedding_computation_time':        { lowerIsBetter: true  }
    };

    const config = metricConfig[metricKey];
    if (!config) return;

    let bestKey   = null;
    let bestValue = config.lowerIsBetter ? Infinity : -Infinity;

    const embeddings = processedData.files[fileKey].embeddings;
    for (const key in embeddings) {
        const metrics = embeddings[key].metrics;
        if (metrics && metrics[metricKey] !== undefined) {
            const value = metrics[metricKey];
            if (config.lowerIsBetter ? value < bestValue : value > bestValue) {
                bestValue = value;
                bestKey   = key;
            }
        }
    }

    if (bestKey) {
        const p = parseEmbeddingKey(bestKey);
        modelSlider.value            = params.model.indexOf(p.model);
        chunkSizeSlider.value        = params.cs.indexOf(p.cs);
        chunkOverlapSlider.value     = params.co.indexOf(p.co);
        themeSlider.value            = params.t.indexOf(p.t);
        chunkingStrategySlider.value = params.s.indexOf(p.s);
        similarityMetricSlider.value = params.m.indexOf(p.m);
        updateView(fileKey, false, metricKey);
    }
}

// ---------------------------------------------------------------------------
// Main view update
// ---------------------------------------------------------------------------
function updateView(fileKey, repopulate = false, highlightedMetric = null) {
    if (repopulate) populateAndSetupSliders(fileKey);

    clearAllDisplays();

    const fileData = processedData.files[fileKey];
    if (!fileData) {
        fileNameSpan.textContent = 'No data for this file.';
        showEmptyState('No data available for this file.');
        return;
    }

    fileNameSpan.textContent = `${fileKey}${fileData.fileName ? '  —  ' + fileData.fileName : ''}`;
    filterEmbeddings();

    if (filteredEmbeddingKeys.length === 0) {
        showEmptyState('No data for this parameter combination.');
        const chunkCountValueEL = document.getElementById('chunk-count-value');
        if (chunkCountValueEL) chunkCountValueEL.textContent = '-';
        return;
    }

    const embeddingKey  = filteredEmbeddingKeys[0];
    const embeddingData = fileData.embeddings[embeddingKey];
    if (!embeddingData) {
        showEmptyState('Error: Embedding data not found.');
        const chunkCountValueEL = document.getElementById('chunk-count-value');
        if (chunkCountValueEL) chunkCountValueEL.textContent = '-';
        return;
    }

    const chunkCountValueEL = document.getElementById('chunk-count-value');
    if (chunkCountValueEL) {
        chunkCountValueEL.textContent = (embeddingData.phrases && embeddingData.phrases.length > 0) ? embeddingData.phrases.length : '-';
    }

    const hasMetrics    = embeddingData.metrics && Object.keys(embeddingData.metrics).length > 0;
    const hasHeatmap    = embeddingData.phrases   && embeddingData.similarities &&
                          embeddingData.phrases.length > 0;
    const hasScatter    = embeddingData.scatter_plot_data;
    
    let plotDataToPass = null;
    let finalHasScatter = false;
    if (hasScatter && typeof embeddingData.scatter_plot_data === 'object') {
        if (embeddingData.scatter_plot_data.x) {
            plotDataToPass = embeddingData.scatter_plot_data;
            finalHasScatter = true;
        } else if (embeddingData.scatter_plot_data[activeProjection] && embeddingData.scatter_plot_data[activeProjection].x) {
            plotDataToPass = embeddingData.scatter_plot_data[activeProjection];
            finalHasScatter = true;
        } else if (embeddingData.scatter_plot_data['umap']) {
            plotDataToPass = embeddingData.scatter_plot_data['umap'];
            finalHasScatter = true;
        } else if (embeddingData.scatter_plot_data['tsne']) {
            plotDataToPass = embeddingData.scatter_plot_data['tsne'];
            finalHasScatter = true;
        }
    }

    if (hasMetrics && hasHeatmap) {
        updateMetrics(embeddingData.metrics, highlightedMetric);
        updateHeatmap(embeddingData.phrases, embeddingData.similarities);
        updateScatterPlot(finalHasScatter ? plotDataToPass : null, activeProjection, embeddingData.phrases);
    } else {
        const missing = [];
        if (!hasMetrics) missing.push('metrics');
        if (!hasHeatmap) missing.push('heatmap data');
        showEmptyState(`Incomplete data. Missing: ${missing.join(', ')}.`);
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------
let scatterChart = null;

function clearAllDisplays() {
    metricsGrid.innerHTML        = '';
    heatmapContainer.innerHTML   = '';
    fileLinksContainer.innerHTML = '';
    if (scatterChart) { scatterChart.destroy(); scatterChart = null; }
    scatterPlotContainer.innerHTML = '';
}

function showEmptyState(message) {
    const html = `<div style="padding:20px;text-align:center;color:#94a3b8;font-size:0.85em;">${message}</div>`;
    metricsGrid.innerHTML          = html;
    heatmapContainer.innerHTML     = html;
    scatterPlotContainer.innerHTML = html;
}

// ---- Metrics ----
function updateMetrics(metrics, highlightedMetric = null) {
    metricsGrid.innerHTML = '';
    if (!metrics || Object.keys(metrics).length === 0) {
        metricsGrid.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:0.85em;">No metrics available.</div>';
        return;
    }

    for (const [key, value] of Object.entries(metrics)) {
        if (value === null || value === undefined) continue;

        const item     = document.createElement('div');
        item.className = 'metric-item';
        item.title     = metricTooltips[key] || '';

        const colorInfo = getMetricColor(key, value);
        // Use the color as the left border accent only — keep bg neutral
        item.style.borderLeftColor = colorInfo.rgb;

        const valueEl    = document.createElement('div');
        valueEl.className = 'value';
        valueEl.textContent = typeof value === 'number' ? value.toFixed(4) : value;

        const labelEl    = document.createElement('div');
        labelEl.className = 'label';
        labelEl.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        const bestBtn    = document.createElement('button');
        bestBtn.className = 'metric-best-btn' + (key === highlightedMetric ? ' active' : '');
        bestBtn.textContent = '▲';
        bestBtn.title    = `Find best combination for: ${key.replace(/_/g, ' ')}`;
        bestBtn.onclick  = () => findBestAndApply(key);

        item.appendChild(valueEl);
        item.appendChild(labelEl);
        item.appendChild(bestBtn);
        metricsGrid.appendChild(item);
    }
}

// ---- Heatmap ----
function updateHeatmap(phrases, similarities) {
    heatmapContainer.innerHTML = '';
    if (!phrases || !similarities || phrases.length === 0) {
        heatmapContainer.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:0.85em;">No heatmap data available.</div>';
        return;
    }
    if (phrases.length !== similarities.length) {
        heatmapContainer.innerHTML = '<div style="padding:20px;text-align:center;color:#b91c1c;font-size:0.85em;">Data error: phrase/similarity length mismatch.</div>';
        return;
    }

    const content = document.createElement('p');
    phrases.forEach((phrase, index) => {
        const score     = similarities[index] || 0.0;
        const colorInfo = cmap_heatmap(score);
        const span      = document.createElement('span');
        span.className = 'chunk';
        span.dataset.similarity = (typeof score === 'number') ? score.toFixed(2) : '0.00';
        span.style.backgroundColor = colorInfo.rgb;
        span.style.color           = colorInfo.isDark ? '#ffffff' : '#1a2332';
        span.textContent           = phrase;
        span.title                 = `Similarity: ${score.toFixed(3)}`;
        content.appendChild(span);
    });
    heatmapContainer.appendChild(content);
    // Ensure heatmap respects the current global threshold (if any)
    if (typeof reapplyHeatmapThreshold === 'function') {
        try { reapplyHeatmapThreshold(currentThreshold || 0.0); } catch (e) { /* tolerate */ }
    }
}

// ---- Scatter plot ----

// Store the current plot data so threshold can be reapplied without rebuilding t-SNE
let currentPlotData = null;
let currentPhrases = null;
let currentThreshold = 0.0;

/**
 * Build single dataset from raw plot data at a given threshold.
 */
function _buildDatasetsAtThreshold(plotData, threshold, phrases) {
    const dataPoints = [];
    const bgColors = [];
    const borderColors = [];

    for (let i = 0; i < plotData.similarities.length; i++) {
        const sim = plotData.similarities[i];
        const phrase = (phrases && phrases[i]) ? phrases[i] : null;
        
        // chunk info: truncated text
        let truncated = "";
        if (phrase) {
            truncated = phrase.substring(0, 300);
            if (phrase.length > 300) truncated += '...';
        }

        dataPoints.push({ 
            x: plotData.x[i], 
            y: plotData.y[i], 
            similarity: sim,
            chunkText: truncated
        });
        
        if (sim >= threshold) {
            const colorInfo = cmap_heatmap(sim);
            const rgba = colorInfo.rgb.replace('rgb', 'rgba').replace(')', ', 0.72)');
            bgColors.push(rgba);
            borderColors.push(colorInfo.rgb);
        } else {
            // Instead of turning points uniformly gray, desaturate / blend them towards white
            const colorInfo = cmap_heatmap(sim);
            // parse rgb(r,g,b)
            const m = colorInfo.rgb.match(/rgb\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)/);
            if (m) {
                const r = parseInt(m[1], 10), g = parseInt(m[2], 10), b = parseInt(m[3], 10);
                const blend = 0.75; // how much to move towards white (0..1)
                const dr = Math.round(r + (255 - r) * blend);
                const dg = Math.round(g + (255 - g) * blend);
                const db = Math.round(b + (255 - b) * blend);
                bgColors.push(`rgba(${dr}, ${dg}, ${db}, 0.40)`);
                borderColors.push(`rgb(${Math.round(r + (255 - r) * 0.33)}, ${Math.round(g + (255 - g) * 0.33)}, ${Math.round(b + (255 - b) * 0.33)})`);
            } else {
                bgColors.push('rgba(220,220,220,0.6)');
                borderColors.push('rgba(200,200,200,0.8)');
            }
        }
    }

        return [
                { data: dataPoints,
                    backgroundColor: bgColors,
                    borderColor:     borderColors,
                    borderWidth: 1, pointRadius: 4, pointHoverRadius: 6 }
        ];
}

/**
 * Re-classify existing chart points when only the threshold changes.
 * Avoids destroying/recreating the chart → instant, reactive update.
 */
function reapplyThreshold(threshold) {
    if (!scatterChart || !currentPlotData) return;
    const newDatasets = _buildDatasetsAtThreshold(currentPlotData, threshold, currentPhrases);
    scatterChart.data.datasets[0].backgroundColor = newDatasets[0].backgroundColor;
    scatterChart.data.datasets[0].borderColor = newDatasets[0].borderColor;
    scatterChart.update('none');  // 'none' = no animation
}

// Apply threshold to textual heatmap: dim chunks with similarity < threshold
function reapplyHeatmapThreshold(threshold) {
    try {
        const chunks = document.querySelectorAll('#heatmap-container .chunk');
        chunks.forEach(el => {
            const sim = parseFloat(el.dataset.similarity || 0);
            if (Number.isNaN(sim)) return;
            if (sim < threshold) {
                el.classList.add('chunk--dimmed');
            } else {
                el.classList.remove('chunk--dimmed');
            }
        });
    } catch (e) {
        // tolerate environments where DOM isn't ready
    }
}

/** Initialise (or update) the threshold slider for the active plot. */
function _initThresholdControl(threshold) {
    const control = document.getElementById('threshold-control');
    const slider  = document.getElementById('threshold-slider');
    const display = document.getElementById('threshold-value-display');
    if (!control || !slider || !display) return;

    control.style.display = 'flex';
    slider.value          = threshold;
    display.textContent   = threshold.toFixed(2);

    // Replace listener to avoid stale closures
    const newSlider = slider.cloneNode(true);
    slider.parentNode.replaceChild(newSlider, slider);
    newSlider.value = threshold;
    newSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        currentThreshold = val;
        document.getElementById('threshold-value-display').textContent = val.toFixed(2);
        reapplyThreshold(val);
        if (typeof reapplyHeatmapThreshold === 'function') reapplyHeatmapThreshold(val);
        if (typeof _syncFloatingThresholdUI === 'function') _syncFloatingThresholdUI(val);
    });
}

// Synchronize threshold UI across the original slider and the floating slider
function _syncFloatingThresholdUI(threshold) {
    const floatSlider = document.getElementById('floating-threshold-slider');
    const floatDisplay = document.getElementById('floating-threshold-value');
    const mainDisplay = document.getElementById('threshold-value-display');
    const mainSlider = document.getElementById('threshold-slider');
    if (floatSlider) floatSlider.value = threshold;
    if (floatDisplay) floatDisplay.textContent = threshold.toFixed(2);
    if (mainSlider) mainSlider.value = threshold;
    if (mainDisplay) mainDisplay.textContent = threshold.toFixed(2);
}

// Initialize floating threshold control: drag, persistence, and event wiring
function initFloatingThreshold() {
    const container = document.getElementById('floating-threshold');
    const hdr = document.getElementById('floating-threshold-hdr');
    const slider = document.getElementById('floating-threshold-slider');
    const display = document.getElementById('floating-threshold-value');
    if (!container || !hdr || !slider || !display) return;

    // restore position
    try {
        const raw = localStorage.getItem('floatingThresholdPos');
        if (raw) {
            const pos = JSON.parse(raw);
            if (typeof pos.left === 'number' && typeof pos.top === 'number') {
                container.style.left = pos.left + 'px';
                container.style.top = pos.top + 'px';
                container.style.transform = 'translateY(0)';
            }
        }
    } catch (e) { /* ignore */ }

    // Dragging
    let dragging = false;
    let startX = 0, startY = 0, origLeft = 0, origTop = 0;
    hdr.addEventListener('mousedown', (ev) => {
        ev.preventDefault();
        dragging = true;
        startX = ev.clientX; startY = ev.clientY;
        origLeft = container.getBoundingClientRect().left; origTop = container.getBoundingClientRect().top;
        document.body.style.userSelect = 'none';
    });
    window.addEventListener('mousemove', (ev) => {
        if (!dragging) return;
        const dx = ev.clientX - startX; const dy = ev.clientY - startY;
        let nx = origLeft + dx; let ny = origTop + dy;
        // constrain
        nx = Math.max(6, Math.min(window.innerWidth - container.offsetWidth - 6, nx));
        ny = Math.max(6, Math.min(window.innerHeight - container.offsetHeight - 6, ny));
        container.style.left = nx + 'px'; container.style.top = ny + 'px'; container.style.transform = 'translateY(0)';
    });
    window.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false; document.body.style.userSelect = '';
        // persist
        try {
            const rect = container.getBoundingClientRect();
            localStorage.setItem('floatingThresholdPos', JSON.stringify({ left: Math.round(rect.left), top: Math.round(rect.top) }));
        } catch (e) { /* ignore */ }
    });

    // touch support
    hdr.addEventListener('touchstart', (ev) => { ev.preventDefault(); const t = ev.touches[0]; hdr.dispatchEvent(new MouseEvent('mousedown', { clientX: t.clientX, clientY: t.clientY })); });
    window.addEventListener('touchmove', (ev) => { const t = ev.touches[0]; window.dispatchEvent(new MouseEvent('mousemove', { clientX: t.clientX, clientY: t.clientY })); });
    window.addEventListener('touchend', () => { window.dispatchEvent(new MouseEvent('mouseup')); });

    // slider interaction → update global threshold and sync
    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        currentThreshold = val;
        display.textContent = val.toFixed(2);
        // sync main UI if present
        _syncFloatingThresholdUI(val);
        reapplyThreshold(val);
        if (typeof reapplyHeatmapThreshold === 'function') reapplyHeatmapThreshold(val);
    });

    // ensure initial sync
    _syncFloatingThresholdUI(currentThreshold || 0.0);
}

function updateScatterPlot(plotData, projName = "tsne", phrases = []) {
    if (scatterChart) { scatterChart.destroy(); scatterChart = null; }
    scatterPlotContainer.innerHTML = '';

    // Hide threshold and metadata control when no data
    const control = document.getElementById('threshold-control');
    const metadata = document.getElementById('tsne-metadata');
    const staticParams = document.getElementById('tsne-static-params');
    
    if (control) control.style.display = 'none';
    if (metadata) metadata.style.display = 'none';
    if (staticParams) staticParams.style.display = 'none';

    if (!plotData || !plotData.x || plotData.x.length === 0) {
        scatterPlotContainer.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:0.85em;">No scatter plot data available.</div>';
        return;
    }

    currentPlotData = plotData;
    currentPhrases = phrases;
    const threshold = currentThreshold;

    const canvas = document.createElement('canvas');
    scatterPlotContainer.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: _buildDatasetsAtThreshold(plotData, threshold, phrases) },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                title: {
                    display: false
                },
                legend: { display: false, position: 'top', labels: { font: { size: 12 }, boxWidth: 12 } },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const raw = ctx.raw;
                            const lines = [];
                            if (raw.chunkText) {
                                const text = `Similarity: ${raw.similarity.toFixed(3)} -> ${raw.chunkText}`;
                                const maxLineLen = 110;
                                let currentLine = '';
                                const words = text.split(' ');
                                for (let w of words) {
                                    if ((currentLine + w).length > maxLineLen) {
                                        if (currentLine) lines.push(currentLine.trim());
                                        currentLine = w + ' ';
                                    } else {
                                        currentLine += w + ' ';
                                    }
                                }
                                if (currentLine) lines.push(currentLine.trim());
                            } else {
                                lines.push(`Similarity: ${raw.similarity.toFixed(3)}`);
                            }
                            return lines;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: projName.toUpperCase() + ' Component 1', font: { size: 11 }, color: '#64748b' },
                    grid:  { color: '#e5e9ef' },
                    ticks: { color: '#64748b', font: { size: 11 } }
                },
                y: {
                    title: { display: true, text: projName.toUpperCase() + ' Component 2', font: { size: 11 }, color: '#64748b' },
                    grid:  { color: '#e5e9ef' },
                    ticks: { color: '#64748b', font: { size: 11 } }
                }
            }
        }
    });

    // Show and configure threshold slider
    _initThresholdControl(threshold);

    // Update Metadata badges and static parameters
    if (metadata) {
        metadata.style.display = 'flex';
        metadata.innerHTML = '';
        
        if (projName === 'tsne') {
            const klValue = typeof plotData.kl_divergence === 'number' ? plotData.kl_divergence.toFixed(3) : (plotData.kl_divergence || 'N/A');
            let colorBorder = '#e2e8f0';
            if (typeof plotData.kl_divergence === 'number') {
                colorBorder = getTSNEColor('kl_divergence', plotData.kl_divergence).rgb;
            }
            
            metadata.innerHTML = `
                <div title="Kullback-Leibler Divergence: Measures how well the 2D projection preserves the original high-dimensional distances. Closer to 0.0 is better." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-left: 3px solid ${colorBorder}; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">KL Div:</span>
                    <span>${klValue}</span>
                </div>
                <div title="Perplexity: The balance between local attention in the algorithm." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">Perplexity:</span>
                    <span>${plotData.perplexity || 'N/A'}</span>
                </div>
                <div title="Iterations: The number of optimization steps." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">Iter:</span>
                    <span>${plotData.n_iter || 'N/A'}</span>
                </div>
            `;
            
            if (staticParams) {
                staticParams.style.display = 'block';
                staticParams.innerHTML = `Algorithm Parameters: 
                    <span style="font-weight: 500;">init</span>="${plotData.init || 'pca'}" &bull; 
                    <span style="font-weight: 500;">learning_rate</span>="${plotData.learning_rate || 'auto'}" &bull; 
                    <span style="font-weight: 500;">max_iter</span>=${plotData.max_iter || '1000'}`;
            }
        } else if (projName === 'umap') {
            metadata.innerHTML = `
                <div title="Number of Neighbors: The size of local neighborhood used for manifold approximation." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">n_neighbors:</span>
                    <span>${plotData.n_neighbors || 'N/A'}</span>
                </div>
                <div title="Minimum Distance: The effective minimum distance between embedded points." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">min_dist:</span>
                    <span>${plotData.min_dist || 'N/A'}</span>
                </div>
            `;
            
            if (staticParams) {
                staticParams.style.display = 'block';
                staticParams.innerHTML = `Algorithm Parameters: 
                    <span style="font-weight: 500;">metric</span>="${plotData.metric || 'cosine'}"`;
            }
        } else if (projName === 'pca') {
            const ev1 = typeof plotData.explained_variance_1 === 'number' ? (plotData.explained_variance_1 * 100).toFixed(2) + '%' : 'N/A';
            const ev2 = typeof plotData.explained_variance_2 === 'number' ? (plotData.explained_variance_2 * 100).toFixed(2) + '%' : 'N/A';
            
            metadata.innerHTML = `
                <div title="Explained Variance: Amount of variance explained by the first principal component." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">Var C1:</span>
                    <span>${ev1}</span>
                </div>
                <div title="Explained Variance: Amount of variance explained by the second principal component." 
                    style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 4px 10px; font-size: 0.75em; color: #475569; cursor: help;">
                    <span style="font-weight: 600; margin-right: 4px;">Var C2:</span>
                    <span>${ev2}</span>
                </div>
            `;
            
            if (staticParams) {
                staticParams.style.display = 'none';
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bootstrap: decode data, then initialise
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    const loadingIndicator = document.getElementById('loading-indicator');
    const mainContainer    = document.querySelector('.container');

    if (!window.Worker) {
        loadingIndicator.innerHTML = '<h2>Browser not supported</h2><p>Web Workers are required. Please use a modern browser.</p>';
        return;
    }

    const b64Data = b64DataChunks.join('');
    const blob    = new Blob([workerScript], { type: 'application/javascript' });
    const worker  = new Worker(URL.createObjectURL(blob));

    worker.onmessage = function (event) {
        if (event.data.success) {
            processedData = event.data.data;
            if (processedData && processedData.files) {
                loadingIndicator.style.display = 'none';
                mainContainer.style.visibility = 'visible';
                initialize();
            } else {
                loadingIndicator.innerHTML = '<h2>Invalid Data</h2><p>The report data structure is missing or malformed.</p>';
            }
        } else {
            console.error('Worker error:', event.data.error);
            loadingIndicator.innerHTML = `<h2>Decoding Error</h2><p>Failed to decompress report data.</p><pre style="font-size:0.8em;color:#b91c1c">${event.data.error}</pre>`;
        }
        URL.revokeObjectURL(worker.objectURL);
    };

    worker.onerror = function (error) {
        console.error('Worker failed:', error);
        loadingIndicator.innerHTML = '<h2>Critical Error</h2><p>The data processing worker encountered a fatal error.</p>';
    };

    worker.postMessage(b64Data);
});

function getCurrentEmbeddingData() {
    return _findOptimalOrFirstValidData(filteredEmbeddingKeys);
}

// Initialize floating control once DOM is ready and data loaded
document.addEventListener('DOMContentLoaded', () => {
    try { initFloatingThreshold(); } catch (e) { /* tolerate */ }
});
