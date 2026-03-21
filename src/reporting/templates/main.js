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

    return cmap_metrics(normalizedScore);
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
        return;
    }

    const embeddingKey  = filteredEmbeddingKeys[0];
    const embeddingData = fileData.embeddings[embeddingKey];
    if (!embeddingData) {
        showEmptyState('Error: Embedding data not found.');
        return;
    }

    const hasMetrics    = embeddingData.metrics && Object.keys(embeddingData.metrics).length > 0;
    const hasHeatmap    = embeddingData.phrases   && embeddingData.similarities &&
                          embeddingData.phrases.length > 0;
    const hasScatter    = embeddingData.scatter_plot_data &&
                          embeddingData.scatter_plot_data.x &&
                          embeddingData.scatter_plot_data.x.length > 0;

    if (hasMetrics && hasHeatmap) {
        updateMetrics(embeddingData.metrics, highlightedMetric);
        updateHeatmap(embeddingData.phrases, embeddingData.similarities);
        updateScatterPlot(hasScatter ? embeddingData.scatter_plot_data : null);
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
        span.style.backgroundColor = colorInfo.rgb;
        span.style.color           = colorInfo.isDark ? '#ffffff' : '#1a2332';
        span.textContent           = phrase;
        span.title                 = `Similarity: ${score.toFixed(3)}`;
        content.appendChild(span);
    });
    heatmapContainer.appendChild(content);
}

// ---- Scatter plot ----

// Store the current plot data so threshold can be reapplied without rebuilding t-SNE
let currentPlotData = null;
let currentThreshold = 0.0;

/**
 * Build single dataset from raw plot data at a given threshold.
 */
function _buildDatasetsAtThreshold(plotData, threshold) {
    const dataPoints = [];
    const bgColors = [];
    const borderColors = [];

    for (let i = 0; i < plotData.similarities.length; i++) {
        const sim = plotData.similarities[i];
        dataPoints.push({ x: plotData.x[i], y: plotData.y[i], similarity: sim });
        
        if (sim >= threshold) {
            const colorInfo = cmap_heatmap(sim);
            const rgba = colorInfo.rgb.replace('rgb', 'rgba').replace(')', ', 0.72)');
            bgColors.push(rgba);
            borderColors.push(colorInfo.rgb);
        } else {
            bgColors.push('rgba(160, 160, 160, 0.5)'); // Gray
            borderColors.push('rgba(100, 100, 100, 0.8)');
        }
    }

    return [
        { label: 'Chunks', data: dataPoints,
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
    const newDatasets = _buildDatasetsAtThreshold(currentPlotData, threshold);
    scatterChart.data.datasets[0].backgroundColor = newDatasets[0].backgroundColor;
    scatterChart.data.datasets[0].borderColor = newDatasets[0].borderColor;
    scatterChart.update('none');  // 'none' = no animation
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
    });
}

function updateScatterPlot(plotData) {
    if (scatterChart) { scatterChart.destroy(); scatterChart = null; }
    scatterPlotContainer.innerHTML = '';

    // Hide threshold control when no data
    const control = document.getElementById('threshold-control');
    if (control) control.style.display = 'none';

    if (!plotData || !plotData.x || plotData.x.length === 0) {
        scatterPlotContainer.innerHTML = '<div style="padding:20px;text-align:center;color:#94a3b8;font-size:0.85em;">No scatter plot data available.</div>';
        return;
    }

    currentPlotData = plotData;
    const threshold = currentThreshold;

    const canvas = document.createElement('canvas');
    scatterPlotContainer.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: _buildDatasetsAtThreshold(plotData, threshold) },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                title: {
                    display: true,
                    text: plotData.title || 't-SNE Visualisation',
                    font: { size: 13, weight: '600' },
                    color: '#1a2332',
                    padding: { bottom: 12 }
                },
                legend: { position: 'top', labels: { font: { size: 12 }, boxWidth: 12 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Similarity: ${ctx.raw.similarity.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 't-SNE Dimension 1', font: { size: 11 }, color: '#64748b' },
                    grid:  { color: '#e5e9ef' },
                    ticks: { color: '#64748b', font: { size: 11 } }
                },
                y: {
                    title: { display: true, text: 't-SNE Dimension 2', font: { size: 11 }, color: '#64748b' },
                    grid:  { color: '#e5e9ef' },
                    ticks: { color: '#64748b', font: { size: 11 } }
                }
            }
        }
    });

    // Show and configure threshold slider
    _initThresholdControl(threshold);
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
