import base64
import json
import os
import zlib
from typing import Any

import numpy as np
from csscompressor import compress as cssmin


def safe_numpy_converter(obj: Any) -> Any:
    """
    Convertit récursivement les types NumPy en types Python natifs pour la sérialisation JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(
        obj,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    if isinstance(obj, dict):
        return {k: safe_numpy_converter(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_numpy_converter(i) for i in obj]
    return obj


def generate_main_page(
    processed_data: dict,
    output_dir: str,
    total_combinations: int,
    single_file: bool = False,
    graph_paths: dict | None = None,
):
    """
    Generates the main interactive web page for heatmap visualization.
    By default, creates one HTML file per markdown file.
    If single_file is True, creates a single index.html for all files.
    """
    graph_paths = graph_paths or {}
    generation_jobs = []

    if single_file:
        job = {
            "data": processed_data,
            "filename": "index.html",
            "graphs": graph_paths.get("global", []),
        }
        generation_jobs.append(job)
    else:
        for file_key, file_data in processed_data["files"].items():
            base_name = os.path.splitext(file_key)[0]
            job_data = {"files": {file_key: file_data}}
            job = {
                "data": job_data,
                "filename": f"{base_name}.html",
                "graphs": graph_paths.get(file_key, []),
            }
            generation_jobs.append(job)

    for job in generation_jobs:
        # Appliquer une conversion profonde pour garantir l'absence de types NumPy
        safe_data = safe_numpy_converter(job["data"])

        # Sérialiser avec JSON, compresser avec zlib, puis encoder en Base64
        json_string = json.dumps(safe_data)
        packed_data = json_string.encode("utf-8")
        compressed_data = zlib.compress(packed_data, level=9)
        b64_data = base64.b64encode(compressed_data).decode("ascii")

        # Chunk the base64 string to avoid browser limits on string literal size
        chunk_size = 50000  # Use 50KB chunks
        b64_chunks = [
            b64_data[i : i + chunk_size] for i in range(0, len(b64_data), chunk_size)
        ]
        js_data_array = "const b64DataChunks = " + json.dumps(b64_chunks) + ";\n"

        css_content = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f9;
            color: #333;
        }
        .container {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            border: 1px solid #d1d9e0;
            border-radius: 8px;
            padding: 5px 10px;
            background-color: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            position: sticky;
            top: 20px;
            z-index: 1000;
        }
        .metrics, .visualization {
            border: 1px solid #d1d9e0;
            border-radius: 8px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .controls h2, .metrics h2, .visualization h2 {
            margin-top: 0;
            color: #34495e;
            border-bottom: 2px solid #e0e6ed;
            padding-bottom: 10px;
        }
        .control-group {
            margin-bottom: 0;
            overflow: hidden;
            transition: filter 0.3s ease;
        }
        .label-group {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
        }
        .control-group label {
            font-weight: bold;
            font-size: 0.9em;
            display: flex;
            align-items: center;
        }
        .label-text {
            white-space: nowrap;
            margin-right: 0.5em;
        }
        .slider-container {
            flex-grow: 1;
            display: flex;
            align-items: center;
        }
        input[type="range"] {
            flex-grow: 1;
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            height: 8px;
            background: #d3d3d3;
            outline: none;
            opacity: 0.9;
            transition: opacity .2s;
            border-radius: 5px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #3498db;
            cursor: pointer;
            border-radius: 50%;
            border: 2px solid #fff;
        }
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: #3498db;
            cursor: pointer;
            border-radius: 50%;
            border: 2px solid #fff;
        }
        input[type="range"]:disabled::-webkit-slider-thumb {
            background: #bdc3c7;
        }
        input[type="range"]:disabled::-moz-range-thumb {
            background: #bdc3c7;
        }
        input[type="range"]:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .label-value {
            font-weight: bold;
            color: #2980b9;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: right;
            flex-grow: 1;
            min-width: 0;
        }
        .metrics {
            flex-shrink: 0;
        }
        .metrics-grid {
            display: flex;
            flex-direction: row;
            gap: 15px;
            overflow-x: auto;
            padding-bottom: 10px;
        }
        .metric-item {
            position: relative;
            flex: 0 0 100px;
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-best-btn {
            position: absolute;
            bottom: 5px;
            right: 5px;
            background: rgba(0,0,0,0.1);
            border: none;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            font-size: 16px;
            line-height: 24px;
            text-align: center;
            cursor: pointer;
            transition: background 0.2s;
        }
        .metric-best-btn:hover {
            background: rgba(0,0,0,0.3);
        }
        .metric-best-btn.active {
            background-color: #3498db;
            color: white;
        }
        .metric-item .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-item .label {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        .visualization {
            overflow-y: auto;
            line-height: 1.8;
        }
        .heatmap-content span {
            padding: 2px 1px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.85em;
            line-height: 1.2;
        }
        .file-links-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .file-links-grid a {
            display: block;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            text-align: center;
            color: #3498db;
            text-decoration: none;
            transition: background-color 0.2s;
        }
        .file-links-grid a:hover {
            background-color: #d1d9e0;
        }
        #scatter-plot-container {
            width: 100%;
            height: 600px;
        }
        """
        minified_css = cssmin(css_content)

        # Contenu du worker JS, maintenant intégré pour un fichier autonome
        worker_js_content = r"""// Import the pako library for zlib decompression
self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js');

self.onmessage = function(event) {
    const base64String = event.data;
    try {
        // Step 1: Decode Base64 to a binary string
        const binaryString = atob(base64String);

        // Step 2: Convert the binary string to a Uint8Array to handle raw bytes
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Step 3: Decompress the byte array using pako
        const decompressedBytes = pako.inflate(bytes);

        // Step 4: Decode the decompressed bytes from UTF-8 into a string
        const jsonString = new TextDecoder("utf-8").decode(decompressedBytes);
        
        // Step 5: Parse the JSON string
        const processedData = JSON.parse(jsonString);
        
        // Send the processed data back to the main thread
        self.postMessage({ success: true, data: processedData });
    } catch (e) {
        // Report an error back to the main thread
        self.postMessage({ success: false, error: e.message });
    }
};"""

        # Generate HTML for graph links
        graph_links_html = ""
        if job.get("graphs"):
            for path in job["graphs"]:
                file_name = os.path.basename(path)
                graph_links_html += (
                    f'<a href="{file_name}" target="_blank">{file_name}</a>'
                )

        # Isoler le code JavaScript statique pour la minification
        static_js_content = r"""
        let processedData = {};

        const metricTooltips = {
            'internal_coherence_score': "Internal Coherence Score (ICS). Measures stability and predictability of similarity measurements. LOWER IS BETTER. (< 0.1: Excellent, 0.1-0.3: Good, 0.3-0.5: Fair, > 0.5: Poor)",
            'local_density_index': "Local Density Index (LDI). Measures if embeddings' nearest neighbors belong to the same theme. HIGHER IS BETTER. (> 0.8: Excellent, 0.6-0.8: Good, 0.4-0.6: Fair, < 0.4: Poor)",
            'robustness_score': "Robustness Score (RS). Tests stability against random noise. HIGHER IS BETTER. (> 0.95: Very Robust, 0.9-0.95: Robust, 0.8-0.9: Moderate, < 0.8: Fragile)",
            'intra_cluster_distance_normalized': "Intra-Cluster Cohesion. Measures how tightly grouped texts of the same theme are. HIGHER IS BETTER. (> 0.8: Excellent, 0.6-0.8: Good, 0.4-0.6: Fair, < 0.4: Poor)",
            'inter_cluster_distance_normalized': "Inter-Cluster Separation. Measures how well different themes are separated. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, < 0.3: Poor)",
            'silhouette_score': "Silhouette Score. Global clustering quality combining cohesion and separation. Range: -1 to 1. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, 0-0.3: Poor, < 0: Very Poor)",
            'mean_similarity': "Mean Similarity. The average cosine similarity between the reference theme embeddings and the document chunk embeddings. HIGHER IS BETTER. Indicates how relevant the document is to the themes on average."
        };

        const fileSlider = document.getElementById('file-slider');
        const fileNameSpan = document.getElementById('file-name');
        const modelSlider = document.getElementById('model-slider');
        const modelNameSpan = document.getElementById('model-name');
        const chunkSizeSlider = document.getElementById('chunk-size-slider');
        const chunkSizeValueSpan = document.getElementById('chunk-size-value');
        const chunkOverlapSlider = document.getElementById('chunk-overlap-slider');
        const chunkOverlapValueSpan = document.getElementById('chunk-overlap-value');
        const themeSlider = document.getElementById('theme-slider');
        const themeNameSpan = document.getElementById('theme-name');
        const chunkingStrategySlider = document.getElementById('chunking-strategy-slider');
        const chunkingStrategyNameSpan = document.getElementById('chunking-strategy-name');
        const similarityMetricSlider = document.getElementById('similarity-metric-slider');
        const similarityMetricNameSpan = document.getElementById('similarity-metric-name');
        const metricsGrid = document.getElementById('metrics-grid');
        const heatmapContainer = document.getElementById('heatmap-container');
        const scatterPlotContainer = document.getElementById('scatter-plot-container');
        const fileLinksContainer = document.getElementById('file-links-container');

        let fileKeys = [];
        let allEmbeddingKeys = [];
        let filteredEmbeddingKeys = [];
        const params = {
            model: [],
            cs: [],
            co: [],
            t: [],
            s: [],
            m: []
        };

        const createCmap = (colorSet) => (score) => {
            if (typeof score !== 'number' || isNaN(score)) score = 0.0;
            score = Math.max(0.0, Math.min(1.0, score));

            // Handle edge case of score being 1 to avoid out-of-bounds access
            if (score === 1.0) {
                const c = colorSet[colorSet.length - 1];
                return {
                    rgb: `rgb(${c.r},${c.g},${c.b})`,
                    isDark: (c.r * 0.299 + c.g * 0.587 + c.b * 0.114) < 128
                };
            }

            const scaledScore = score * (colorSet.length - 1);
            const i = Math.floor(scaledScore);
            const t = scaledScore - i;

            const c1 = colorSet[i];
            const c2 = colorSet[i + 1];

            const r = Math.round(c1.r * (1 - t) + c2.r * t);
            const g = Math.round(c1.g * (1 - t) + c2.g * t);
            const b = Math.round(c1.b * (1 - t) + c2.b * t);

            return {
                rgb: `rgb(${r},${g},${b})`,
                isDark: (r * 0.299 + g * 0.587 + b * 0.114) < 128
            };
        };

        const cmap_heatmap = createCmap([
            { r: 43, g: 131, b: 186 },    // Poor/Low similarity (Blue/Green)
            { r: 171, g: 221, b: 164 },  // Light Green
            { r: 255, g: 255, b: 191 }, // Neutral (Yellow)
            { r: 253, g: 174, b: 97 },   // Orange
            { r: 215, g: 25, b: 28 }    // Excellent/High similarity (Red)
        ]);

        function getMetricColor(metricKey, value) {
            // A tasteful color gradient from Red (bad) -> Yellow (medium) -> Green (good)
            const cmap_metrics = createCmap([
                { r: 215, g: 48, b: 39 },   // Red
                { r: 254, g: 224, b: 144 }, // Yellow
                { r: 26, g: 152, b: 80 }    // Green
            ]);

            // Configuration for each metric: min/max for color scaling, and whether lower is better
            const metricConfigs = {
                'internal_coherence_score': { min: 0.0, max: 0.5, lowerIsBetter: true },
                'local_density_index': { min: 0.4, max: 1.0, lowerIsBetter: false },
                'robustness_score': { min: 0.8, max: 1.0, lowerIsBetter: false },
                'intra_cluster_distance_normalized': { min: 0.4, max: 1.0, lowerIsBetter: false },
                'inter_cluster_distance_normalized': { min: 0.3, max: 0.8, lowerIsBetter: false },
                'silhouette_score': { min: -0.1, max: 0.7, lowerIsBetter: false },
                'mean_similarity': { min: 0.5, max: 1.0, lowerIsBetter: false }
            };

            const config = metricConfigs[metricKey];
            if (!config) {
                return { rgb: '#ecf0f1', isDark: false }; // Neutral for unknown metrics
            }

            // Clamp value to the defined range for color scaling
            const clampedValue = Math.max(config.min, Math.min(config.max, value));

            // Normalize the score to a 0-1 range where 1 is always better
            let normalizedScore;
            if (config.max === config.min) {
                normalizedScore = 0.5; // Neutral if range is zero
            } else if (config.lowerIsBetter) {
                normalizedScore = (config.max - clampedValue) / (config.max - config.min);
            } else {
                normalizedScore = (clampedValue - config.min) / (config.max - config.min);
            }

            return cmap_metrics(normalizedScore);
        }

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

        function populateAndSetupSliders(fileKey) {
            if (!fileKey || !processedData.files[fileKey]) return;

            const currentValues = {
                model: params.model[modelSlider.value],
                cs: params.cs[chunkSizeSlider.value],
                co: params.co[chunkOverlapSlider.value],
                t: params.t[themeSlider.value],
                s: params.s[chunkingStrategySlider.value],
                m: params.m[similarityMetricSlider.value]
            };

            const paramSets = {
                model: new Set(),
                cs: new Set(),
                co: new Set(),
                t: new Set(),
                s: new Set(),
                m: new Set()
            };

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
            params.cs = [...paramSets.cs].sort((a, b) => a - b);
            params.co = [...paramSets.co].sort((a, b) => a - b);
            params.t = [...paramSets.t].sort();
            params.s = [...paramSets.s].sort();
            params.m = [...paramSets.m].sort();

            const setupSlider = (slider, values, previousValue) => {
                slider.max = values.length > 0 ? values.length - 1 : 0;
                slider.disabled = values.length <= 1;
                const newIndex = values.indexOf(previousValue);
                slider.value = newIndex !== -1 ? newIndex : 0;
            };

            setupSlider(modelSlider, params.model, currentValues.model);
            setupSlider(chunkSizeSlider, params.cs, currentValues.cs);
            setupSlider(chunkOverlapSlider, params.co, currentValues.co);
            setupSlider(themeSlider, params.t, currentValues.t);
            setupSlider(chunkingStrategySlider, params.s, currentValues.s);
            setupSlider(similarityMetricSlider, params.m, currentValues.m);
        }

        function initialize() {
            fileKeys = Object.keys(processedData.files || {}).sort();
            if (fileKeys.length === 0) {
                console.error("No files found in processed data.");
                return;
            }

            const setupSlider = (slider, values, previousValue) => {
                slider.max = values.length > 0 ? values.length - 1 : 0;
                slider.disabled = values.length <= 1;
                const newIndex = values.indexOf(previousValue);
                slider.value = newIndex !== -1 ? newIndex : 0;
            };

            setupSlider(fileSlider, fileKeys, fileKeys[0]);

            const initialFileKey = fileKeys[0];
            updateView(initialFileKey, true);

            fileSlider.addEventListener('input', (e) => {
                const fileKey = fileKeys[parseInt(e.target.value, 10)];
                updateView(fileKey, true);
            });

            const otherSliders = [modelSlider, chunkSizeSlider, chunkOverlapSlider, themeSlider, chunkingStrategySlider, similarityMetricSlider];
            otherSliders.forEach(slider => {
                slider.addEventListener('input', () => {
                    const fileKey = fileKeys[parseInt(fileSlider.value, 10)];
                    updateView(fileKey);
                });
            });
        }

        function filterEmbeddings() {
            const selectedModel = params.model[modelSlider.value];
            const selectedCS = params.cs[chunkSizeSlider.value];
            const selectedCO = params.co[chunkOverlapSlider.value];
            const selectedT = params.t[themeSlider.value];
            const selectedS = params.s[chunkingStrategySlider.value];
            const selectedM = params.m[similarityMetricSlider.value];

            modelNameSpan.textContent = selectedModel || 'N/A';
            chunkSizeValueSpan.textContent = selectedCS || 'N/A';
            chunkOverlapValueSpan.textContent = selectedCO || 'N/A';
            themeNameSpan.textContent = selectedT || 'N/A';
            chunkingStrategyNameSpan.textContent = selectedS || 'N/A';
            similarityMetricNameSpan.textContent = selectedM || 'N/A';

            filteredEmbeddingKeys = allEmbeddingKeys.filter(key => {
                const p = parseEmbeddingKey(key);
                return p.model === selectedModel &&
                       p.cs === selectedCS &&
                       p.co === selectedCO &&
                       p.t === selectedT &&
                       p.s === selectedS &&
                       p.m === selectedM;
            });
        }

        function findBestAndApply(metricKey) {
            console.log(`Finding best value for metric: ${metricKey}`);
            const fileKey = fileKeys[parseInt(fileSlider.value, 10)];
            if (!fileKey || !processedData.files[fileKey]) return;

            const metricConfig = {
                'internal_coherence_score': { lowerIsBetter: true },
                'local_density_index': { lowerIsBetter: false },
                'robustness_score': { lowerIsBetter: false },
                'intra_cluster_distance_normalized': { lowerIsBetter: false },
                'inter_cluster_distance_normalized': { lowerIsBetter: false },
                'silhouette_score': { lowerIsBetter: false },
                'mean_similarity': { lowerIsBetter: false }
            };

            const config = metricConfig[metricKey];
            if (!config) return;

            let bestKey = null;
            let bestValue = config.lowerIsBetter ? Infinity : -Infinity;

            const embeddings = processedData.files[fileKey].embeddings;
            for (const key in embeddings) {
                const metrics = embeddings[key].metrics;
                if (metrics && metrics[metricKey] !== undefined) {
                    const value = metrics[metricKey];
                    if (config.lowerIsBetter) {
                        if (value < bestValue) {
                            bestValue = value;
                            bestKey = key;
                        }
                    } else {
                        if (value > bestValue) {
                            bestValue = value;
                            bestKey = key;
                        }
                    }
                }
            }

            if (bestKey) {
                const p = parseEmbeddingKey(bestKey);
                
                modelSlider.value = params.model.indexOf(p.model);
                chunkSizeSlider.value = params.cs.indexOf(p.cs);
                chunkOverlapSlider.value = params.co.indexOf(p.co);
                themeSlider.value = params.t.indexOf(p.t);
                chunkingStrategySlider.value = params.s.indexOf(p.s);
                similarityMetricSlider.value = params.m.indexOf(p.m);
                
                updateView(fileKey, false, metricKey);
            }
        }

        function updateView(fileKey, repopulate = false, highlightedMetric = null) {
            console.log('updateView called with:', fileKey, 'repopulate:', repopulate);
            
            if (repopulate) {
                populateAndSetupSliders(fileKey);
            }

            // Effacer immédiatement tous les conteneurs pour éviter les états incohérents
            clearAllDisplays();

            const fileData = processedData.files[fileKey];
            console.log('fileData:', fileData);
            
            if (!fileData) {
                fileNameSpan.textContent = 'No data for this file.';
                showEmptyState('No data available for this file.');
                return;
            }

            fileNameSpan.textContent = `${fileKey} - ${fileData.fileName || ''}`;
            filterEmbeddings();
            console.log('filteredEmbeddingKeys:', filteredEmbeddingKeys);

            if (filteredEmbeddingKeys.length === 0) {
                showEmptyState('No data for this parameter combination.');
                return;
            }

            const embeddingKey = filteredEmbeddingKeys[0];
            const embeddingData = fileData.embeddings[embeddingKey];
            console.log('embeddingData:', embeddingData);

            if (!embeddingData) {
                showEmptyState('Error: Embedding data not found.');
                return;
            }

            // Vérifier la cohérence des données avant mise à jour
            const hasMetrics = embeddingData.metrics && Object.keys(embeddingData.metrics).length > 0;
            const hasHeatmapData = embeddingData.phrases && embeddingData.similarities && 
                                   embeddingData.phrases.length > 0 && embeddingData.similarities.length > 0;
            const hasScatterData = embeddingData.scatter_plot_data && 
                                   embeddingData.scatter_plot_data.x && 
                                   embeddingData.scatter_plot_data.y && 
                                   embeddingData.scatter_plot_data.x.length > 0;

            console.log('Data availability:', { hasMetrics, hasHeatmapData, hasScatterData });

            // Mise à jour atomique : soit tout, soit rien
            if (hasMetrics && hasHeatmapData) {
                updateMetrics(embeddingData.metrics, highlightedMetric);
                updateHeatmap(embeddingData.phrases, embeddingData.similarities);
                
                if (hasScatterData) {
                    updateScatterPlot(embeddingData.scatter_plot_data);
                } else {
                    updateScatterPlot(null);
                }
            } else {
                // Données incomplètes - afficher un message d'erreur cohérent
                let missingParts = [];
                if (!hasMetrics) missingParts.push('metrics');
                if (!hasHeatmapData) missingParts.push('heatmap data');
                if (!hasScatterData) missingParts.push('scatter plot data');
                
                console.log('Missing data parts:', missingParts);
                showEmptyState(`Incomplete data for this combination. Missing: ${missingParts.join(', ')}.`);
            }
        }

        let scatterChart = null; // Variable to hold the chart instance

        function clearAllDisplays() {
            metricsGrid.innerHTML = '';
            heatmapContainer.innerHTML = '';
            fileLinksContainer.innerHTML = '';
            // Destroy the old chart instance if it exists
            if (scatterChart) {
                scatterChart.destroy();
                scatterChart = null;
            }
            scatterPlotContainer.innerHTML = ''; // Clear the container
        }

        function showEmptyState(message) {
            metricsGrid.innerHTML = `<div style="padding: 20px; text-align: center; color: #666; grid-column: 1 / -1;">${message}</div>`;
            heatmapContainer.innerHTML = `<div style="padding: 20px; text-align: center; color: #666;">${message}</div>`;
            fileLinksContainer.innerHTML = '';
            scatterPlotContainer.innerHTML = `<div style="padding: 20px; text-align: center; color: #666;">${message}</div>`;
        }

        function updateScatterPlot(plotData) {
            if (scatterChart) {
                scatterChart.destroy();
                scatterChart = null;
            }
            scatterPlotContainer.innerHTML = ''; // Clear previous content

            if (!plotData || !plotData.x || !plotData.y || plotData.x.length === 0) {
                scatterPlotContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No scatter plot data available for this selection.</div>';
                return;
            }

            const canvas = document.createElement('canvas');
            scatterPlotContainer.appendChild(canvas);
            const ctx = canvas.getContext('2d');

            const datasets = [];
            const uniqueLabels = [...new Set(plotData.labels)];
            
            const colors = {
                'Above Threshold': 'rgba(255, 99, 132, 0.7)', // red
                'Below Threshold': 'rgba(54, 162, 235, 0.7)'  // blue
            };

            uniqueLabels.forEach(label => {
                const dataset = {
                    label: label,
                    data: [],
                    backgroundColor: colors[label] || 'rgba(128, 128, 128, 0.7)',
                    borderColor: colors[label] ? colors[label].replace('0.7', '1') : 'rgba(128, 128, 128, 1)',
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 7
                };
                
                for (let i = 0; i < plotData.labels.length; i++) {
                    if (plotData.labels[i] === label) {
                        dataset.data.push({
                            x: plotData.x[i],
                            y: plotData.y[i],
                            similarity: plotData.similarities[i]
                        });
                    }
                }
                datasets.push(dataset);
            });

            if (datasets.length === 0) {
                scatterPlotContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No valid data for scatter plot.</div>';
                return;
            }

            scatterChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        title: {
                            display: true,
                            text: plotData.title || 't-SNE Visualization'
                        },
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = context.raw;
                                    return `Similarity: ${point.similarity.toFixed(4)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 't-SNE Dimension 1'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 't-SNE Dimension 2'
                            }
                        }
                    }
                }
            });
        }

        function updateMetrics(metrics, highlightedMetric = null) {
            metricsGrid.innerHTML = '';
            if (!metrics || Object.keys(metrics).length === 0) {
                metricsGrid.innerHTML = '<div style="padding: 20px; text-align: center; color: #666; grid-column: 1 / -1;">No metrics available.</div>';
                return;
            }

            for (const [key, value] of Object.entries(metrics)) {
                if (value === null || value === undefined) continue;
                
                const metricItem = document.createElement('div');
                metricItem.className = 'metric-item';
                metricItem.title = metricTooltips[key] || 'No description available.';
                
                const colorInfo = getMetricColor(key, value);
                metricItem.style.backgroundColor = colorInfo.rgb;
                
                const valueSpan = document.createElement('div');
                valueSpan.className = 'value';
                valueSpan.style.color = colorInfo.isDark ? '#fff' : '#2c3e50';
                valueSpan.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                
                const labelSpan = document.createElement('div');
                labelSpan.className = 'label';
                labelSpan.style.color = colorInfo.isDark ? '#ecf0f1' : '#7f8c8d';
                labelSpan.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                const bestBtn = document.createElement('button');
                bestBtn.className = 'metric-best-btn';
                if (key === highlightedMetric) {
                    bestBtn.classList.add('active');
                }
                bestBtn.innerHTML = '🏆';
                bestBtn.title = `Find best combination for ${key}`;
                bestBtn.onclick = () => findBestAndApply(key);

                metricItem.appendChild(valueSpan);
                metricItem.appendChild(labelSpan);
                metricItem.appendChild(bestBtn);
                metricsGrid.appendChild(metricItem);
            }
        }

        function updateHeatmap(phrases, similarities) {
            heatmapContainer.innerHTML = '';
            if (!phrases || !similarities || phrases.length === 0 || similarities.length === 0) {
                heatmapContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No heatmap data available.</div>';
                return;
            }

            if (phrases.length !== similarities.length) {
                heatmapContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #d32f2f;">Error: Mismatch between phrases and similarities data.</div>';
                return;
            }

            const content = document.createElement('p');

            phrases.forEach((phrase, index) => {
                const score = Math.max(0.0, Math.min(1.0, similarities[index] || 0.0));
                const colorInfo = cmap_heatmap(score);
                
                const span = document.createElement('span');
                span.style.backgroundColor = colorInfo.rgb;
                span.style.color = colorInfo.isDark ? '#fff' : '#333';
                span.textContent = phrase;
                span.title = `Similarity: ${score.toFixed(3)}`;
                
                content.appendChild(span);
            });
            heatmapContainer.appendChild(content);
        }

        document.addEventListener('DOMContentLoaded', function() {
            const loadingIndicator = document.getElementById('loading-indicator');
            const mainContainer = document.querySelector('.container');

            if (!window.Worker) {
                loadingIndicator.innerHTML = '<h1>Error</h1><p>Your browser does not support Web Workers. Please use a modern browser.</p>';
                return;
            }

            // Reconstruct the base64 string from chunks
            const b64Data = b64DataChunks.join('');

            const blob = new Blob([workerScript], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));

            worker.onmessage = function(event) {
                if (event.data.success) {
                    processedData = event.data.data;
                    if (processedData && processedData.files) {
                        loadingIndicator.style.display = 'none';
                        mainContainer.style.visibility = 'visible';
                        initialize();
                    } else {
                        loadingIndicator.innerHTML = '<h1>Error: Invalid Data</h1><p>The processed data is missing or invalid.</p>';
                    }
                } else {
                    console.error('Worker error:', event.data.error);
                    loadingIndicator.innerHTML = `<h1>Error</h1><p>Failed to decode data. The data may be corrupted. Check the console for details.</p><pre>${event.data.error}</pre>`;
                }
                URL.revokeObjectURL(worker.objectURL);
            };

            worker.onerror = function(error) {
                console.error('Worker failed:', error);
                loadingIndicator.innerHTML = '<h1>Error</h1><p>A critical error occurred in the data processing worker.</p>';
            };

            worker.postMessage(b64Data);
        });
        """

        # Combiner les données (en morceaux), le worker et le code statique (non minifié)
        final_js_content = (
            js_data_array
            + f"const workerScript = `{worker_js_content}`;\n"
            + static_js_content
        )

        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <style>
        #loading-indicator {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        .container {{
            visibility: hidden;
        }}
        {minified_css}
    </style>
</head>
<body>
    <div id="loading-indicator">
        <h1>Loading Report Data...</h1>
        <p>This may take a moment for large files.</p>
    </div>

    <div class="container">
        <h1>Interactive Embeddings Analysis</h1>
        <p style="text-align: center; margin-top: -15px; color: #555;">
            Displaying results for {total_combinations} parameter combinations.
        </p>

        <!-- Controls area -->
        <div class="controls">
            <div class="control-group">
                <label for="file-slider"><span class="label-text">Markdown file :</span><span id="file-name" class="label-value"></span></label>
                <input type="range" id="file-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="model-slider"><span class="label-text">Model :</span><span id="model-name" class="label-value"></span></label>
                <input type="range" id="model-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="theme-slider"><span class="label-text">Theme Set :</span><span id="theme-name" class="label-value"></span></label>
                <input type="range" id="theme-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="chunk-size-slider"><span class="label-text">Chunk Size :</span><span id="chunk-size-value" class="label-value"></span></label>
                <input type="range" id="chunk-size-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="chunk-overlap-slider"><span class="label-text">Chunk Overlap :</span><span id="chunk-overlap-value" class="label-value"></span></label>
                <input type="range" id="chunk-overlap-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="chunking-strategy-slider"><span class="label-text">Chunking Strategy :</span><span id="chunking-strategy-name" class="label-value"></span></label>
                <input type="range" id="chunking-strategy-slider" min="0" max="0" value="0">
            </div>
            <div class="control-group">
                <label for="similarity-metric-slider"><span class="label-text">Similarity Metric :</span><span id="similarity-metric-name" class="label-value"></span></label>
                <input type="range" id="similarity-metric-slider" min="0" max="0" value="0">
            </div>
        </div>

        <!-- Metrics area -->
        <div class="metrics">
            <h2>Metrics</h2>
            <div id="metrics-grid" class="metrics-grid"></div>
        </div>

        <!-- Visualization area -->
        <div class="visualization">
            <h2>Textual Heatmap</h2>
            <div id="heatmap-container" class="heatmap-content"></div>
        </div>

        <!-- Links and Plot area -->
        <div class="visualization">
            <h2>Reports and Visualizations</h2>
            <div id="file-links-container" class="file-links-grid">
                {graph_links_html}
            </div>
            <div id="scatter-plot-container"></div>
        </div>
    </div>

    <script>{final_js_content}</script>
</body>
</html>
"""
        output_path = os.path.join(output_dir, job["filename"])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated main page at: {output_path}")
