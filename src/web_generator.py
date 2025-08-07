import base64
import json
import os
import zlib
from typing import Any, Dict

import numpy as np
from cssmin import cssmin
from jsmin import jsmin


def numpy_encoder(o: Any) -> Any:
    """Custom encoder for JSON to handle NumPy data types."""
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.number) and hasattr(o, "dtype") and o.dtype == np.uint16:
        # Handle quantized similarity data
        return float(o) / 65535.0
    else:
        return o


def generate_main_page(
    processed_data: Dict[str, Any],
    output_dir: str,
    total_combinations: int,
    single_file: bool = False,
):
    """
    Generates the main interactive web page for heatmap visualization.
    By default, creates one HTML file per markdown file.
    If single_file is True, creates a single index.html for all files.
    """

    generation_jobs = []
    if single_file:
        generation_jobs.append({"data": processed_data, "filename": "index.html"})
    else:
        for file_key, file_data in processed_data["files"].items():
            base_name = os.path.splitext(file_key)[0]
            job_data = {"files": {file_key: file_data}}
            generation_jobs.append({"data": job_data, "filename": f"{base_name}.html"})

    for job in generation_jobs:
        # Sérialiser avec JSON, compresser avec zlib, puis encoder en Base64
        json_string = json.dumps(job["data"], default=numpy_encoder)
        packed_data = json_string.encode("utf-8")
        compressed_data = zlib.compress(packed_data, level=9)
        b64_data = base64.b64encode(compressed_data).decode("ascii")

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
            flex: 0 0 100px;
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
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

        js_content = f"const b64Data = '{b64_data}';\n"
        js_content += r"""
        let processedData = {};

        // Fonction pour décoder les données Base64, zlib et JSON
        function decodeData(base64String) {
            try {
                const byteCharacters = atob(base64String);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const compressedByteArray = new Uint8Array(byteNumbers);
                const decompressedByteArray = pako.inflate(compressedByteArray);
                // Use a TextDecoder to convert the Uint8Array to a string
                const jsonString = new TextDecoder().decode(decompressedByteArray);
                return JSON.parse(jsonString);
            } catch (e) {
                console.error("Failed to decode data:", e);
                // Afficher une erreur claire à l'utilisateur
                document.body.innerHTML = '<div style="padding: 20px; text-align: center; font-size: 1.2em; color: red;">Error: Could not load report data. The data may be corrupted.</div>';
                return null;
            }
        }

        const metricTooltips = {
            'internal_coherence_score': "Internal Coherence Score (ICS). Measures stability and predictability of similarity measurements. LOWER IS BETTER. (< 0.1: Excellent, 0.1-0.3: Good, 0.3-0.5: Fair, > 0.5: Poor)",
            'local_density_index': "Local Density Index (LDI). Measures if embeddings' nearest neighbors belong to the same theme. HIGHER IS BETTER. (> 0.8: Excellent, 0.6-0.8: Good, 0.4-0.6: Fair, < 0.4: Poor)",
            'robustness_score': "Robustness Score (RS). Tests stability against random noise. HIGHER IS BETTER. (> 0.95: Very Robust, 0.9-0.95: Robust, 0.8-0.9: Moderate, < 0.8: Fragile)",
            'intra_cluster_distance_normalized': "Intra-Cluster Cohesion. Measures how tightly grouped texts of the same theme are. HIGHER IS BETTER. (> 0.8: Excellent, 0.6-0.8: Good, 0.4-0.6: Fair, < 0.4: Poor)",
            'inter_cluster_distance_normalized': "Inter-Cluster Separation. Measures how well different themes are separated. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, < 0.3: Poor)",
            'silhouette_score': "Silhouette Score. Global clustering quality combining cohesion and separation. Range: -1 to 1. HIGHER IS BETTER. (> 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Fair, 0-0.3: Poor, < 0: Very Poor)"
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
            // Ensure score is a number and normalize to [0,1]
            if (typeof score !== 'number' || isNaN(score)) score = 0.0;
            
            // If score appears to be quantized (integer > 1), dequantize it
            if (score > 1.0 && Number.isInteger(score)) {
                score = Math.min(score, 65535) / 65535.0;
            }
            
            // Clamp to [0,1] range
            score = Math.max(0.0, Math.min(1.0, score));
            
            const i = Math.min(Math.floor(score * (colorSet.length - 1)), colorSet.length - 2);
            const t = (score * (colorSet.length - 1)) % 1;
            const r = Math.round(colorSet[i].r * (1 - t) + colorSet[i+1].r * t);
            const g = Math.round(colorSet[i].g * (1 - t) + colorSet[i+1].g * t);
            const b = Math.round(colorSet[i].b * (1 - t) + colorSet[i+1].b * t);
            return {
                rgb: `rgb(${r},${g},${b})`,
                isDark: (r * 0.299 + g * 0.587 + b * 0.114) < 128
            };
        };

        const cmap_heatmap = createCmap([
            { r: 215, g: 25, b: 28 },    // Poor/Low similarity (Red)
            { r: 253, g: 174, b: 97 },   // Orange
            { r: 255, g: 255, b: 191 }, // Neutral (Yellow)
            { r: 171, g: 221, b: 164 },  // Light Green
            { r: 43, g: 131, b: 186 }    // Excellent/High similarity (Blue/Green)
        ]);

        function getMetricColor(metricKey, value) {
            // Configuration for each metric: min, max, and whether good values are high (false) or low (true)
            const metricConfigs = {
                'internal_coherence_score': { 
                    min: 0, 
                    max: 1, 
                    lowerIsBetter: true,  // RED when high, GREEN when low
                    excellentThreshold: 0.1,
                    goodThreshold: 0.3,
                    fairThreshold: 0.5
                },
                'local_density_index': { 
                    min: 0, 
                    max: 1, 
                    lowerIsBetter: false, // GREEN when high, RED when low
                    excellentThreshold: 0.8,
                    goodThreshold: 0.6,
                    fairThreshold: 0.4
                },
                'robustness_score': { 
                    min: 0, 
                    max: 1, 
                    lowerIsBetter: false, // GREEN when high, RED when low
                    excellentThreshold: 0.95,
                    goodThreshold: 0.9,
                    fairThreshold: 0.8
                },
                'intra_cluster_distance_normalized': { 
                    min: 0, 
                    max: 1, 
                    lowerIsBetter: false, // GREEN when high, RED when low
                    excellentThreshold: 0.8,
                    goodThreshold: 0.6,
                    fairThreshold: 0.4
                },
                'inter_cluster_distance_normalized': { 
                    min: 0, 
                    max: 1, 
                    lowerIsBetter: false, // GREEN when high, RED when low
                    excellentThreshold: 0.7,
                    goodThreshold: 0.5,
                    fairThreshold: 0.3
                },
                'silhouette_score': { 
                    min: -1, 
                    max: 1, 
                    lowerIsBetter: false, // GREEN when high, RED when low
                    excellentThreshold: 0.7,
                    goodThreshold: 0.5,
                    fairThreshold: 0.3,
                    poorThreshold: 0.0
                }
            };

            const config = metricConfigs[metricKey];
            if (!config) {
                // Default for unknown metrics
                return cmap_heatmap(0.5);
            }

            // Handle quantized values: if integer > 1, likely quantized
            if (typeof value === 'number' && value > 1 && Number.isInteger(value)) {
                // Dequantize based on metric type
                if (metricKey === 'silhouette_score') {
                    // Silhouette was stored as [0,65535] mapping to [-1,1]
                    value = ((value / 65535.0) * 2.0) - 1.0;
                } else if (metricKey === 'internal_coherence_score') {
                    // Coherence was inverted: 1 - (value/65535)
                    value = 1.0 - (value / 65535.0);
                } else {
                    // Standard [0,1] metrics
                    value = value / 65535.0;
                }
            }

            // Clamp value to expected range
            value = Math.max(config.min, Math.min(config.max, value));

            // Determine quality level based on thresholds
            let qualityScore;
            
            if (config.lowerIsBetter) {
                // For metrics where lower is better (like internal_coherence_score)
                if (value <= config.excellentThreshold) {
                    qualityScore = 1.0; // Excellent
                } else if (value <= config.goodThreshold) {
                    qualityScore = 0.75; // Good
                } else if (value <= config.fairThreshold) {
                    qualityScore = 0.5; // Fair
                } else {
                    qualityScore = 0.0; // Poor
                }
            } else {
                // For metrics where higher is better
                if (metricKey === 'silhouette_score') {
                    // Special handling for silhouette score with negative values
                    if (value >= config.excellentThreshold) {
                        qualityScore = 1.0; // Excellent
                    } else if (value >= config.goodThreshold) {
                        qualityScore = 0.75; // Good
                    } else if (value >= config.fairThreshold) {
                        qualityScore = 0.5; // Fair
                    } else if (value >= (config.poorThreshold || 0)) {
                        qualityScore = 0.25; // Poor
                    } else {
                        qualityScore = 0.0; // Very Poor
                    }
                } else {
                    // Standard higher-is-better metrics
                    if (value >= config.excellentThreshold) {
                        qualityScore = 1.0; // Excellent
                    } else if (value >= config.goodThreshold) {
                        qualityScore = 0.75; // Good
                    } else if (value >= config.fairThreshold) {
                        qualityScore = 0.5; // Fair
                    } else {
                        qualityScore = 0.0; // Poor
                    }
                }
            }

            return cmap_heatmap(qualityScore);
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

        function updateView(fileKey, repopulate = false) {
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

            fileNameSpan.textContent = fileKey;
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
                updateMetrics(embeddingData.metrics);
                updateHeatmap(embeddingData.phrases, embeddingData.similarities);
                updateFileLinks(embeddingKey, fileKey);
                
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

        function updateMetrics(metrics) {
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

                metricItem.appendChild(valueSpan);
                metricItem.appendChild(labelSpan);
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

            // Normalize similarities to ensure proper color mapping
            const normalizedSimilarities = similarities.map(score => {
                if (typeof score !== 'number' || isNaN(score)) return 0.0;
                // If score appears quantized, dequantize it
                if (score > 1.0 && Number.isInteger(score)) {
                    score = Math.min(score, 65535) / 65535.0;
                }
                return Math.max(0.0, Math.min(1.0, score));
            });

            phrases.forEach((phrase, index) => {
                const score = normalizedSimilarities[index];
                const colorInfo = cmap_heatmap(score);
                
                const span = document.createElement('span');
                span.style.backgroundColor = colorInfo.rgb;
                span.style.color = colorInfo.isDark ? '#fff' : '#333';
                span.textContent = phrase + '. ';
                span.title = `Similarity: ${score.toFixed(3)}`;
                
                content.appendChild(span);
            });
            heatmapContainer.appendChild(content);
        }

        function updateFileLinks(embeddingKey, fileKey) {
            // Cette fonction peut être vide si les liens de fichiers ne sont pas utilisés
            fileLinksContainer.innerHTML = '';
        }

        // Fonctions de validation

        function safeGetData(obj, path, defaultValue = null) {
            /**
             * Accès sécurisé aux propriétés imbriquées d'un objet
             * @param {Object} obj - L'objet à parcourir
             * @param {string} path - Le chemin vers la propriété (ex: "files.fileKey.metrics")
             * @param {*} defaultValue - Valeur par défaut si la propriété n'existe pas
             * @returns {*} La valeur trouvée ou la valeur par défaut
             */
            const keys = path.split('.');
            let current = obj;
            
            for (const key of keys) {
                if (current === null || current === undefined || !(key in current)) {
                    return defaultValue;
                }
                current = current[key];
            }
            
            return current;
        }

        function validateFileData(fileData) {
            /**
             * Valide qu'un objet fileData contient les propriétés requises
             * @param {Object} fileData - Les données d'un fichier
             * @returns {Object} Objet avec les propriétés manquantes complétées
             */
            const defaultFileData = {
                phrases: [],
                similarities: [],
                metrics: {
                    silhouette_score: -1,
                    mean_similarity: 0,
                    n_clusters: 0
                },
                scatter_plot_data: null
            };
            
            if (!fileData) {
                return defaultFileData;
            }
            
            return {
                phrases: fileData.phrases || defaultFileData.phrases,
                similarities: fileData.similarities || defaultFileData.similarities,
                metrics: {
                    ...defaultFileData.metrics,
                    ...(fileData.metrics || {})
                },
                scatter_plot_data: fileData.scatter_plot_data || defaultFileData.scatter_plot_data
            };
        }

        function updateScatterPlot(selectedKey) {
            /**
             * Met à jour le graphique scatter plot avec validation des données
             */
            const embeddings = safeGetData(processedData, `files.${selectedKey}.embeddings`, {});
            const embeddingKey = getSelectedEmbeddingKey();
            
            if (!embeddingKey || !(embeddingKey in embeddings)) {
                // Afficher un message d'erreur au lieu de planter
                const scatterDiv = document.getElementById('scatter-plot');
                if (scatterDiv) {
                    scatterDiv.innerHTML = '<p>Aucune donnée de scatter plot disponible pour cette configuration.</p>';
                }
                return;
            }
            
            const fileData = validateFileData(embeddings[embeddingKey]);
            const scatterData = fileData.scatter_plot_data;
            
            if (!scatterData || !scatterData.x || !scatterData.y) {
                const scatterDiv = document.getElementById('scatter-plot');
                if (scatterDiv) {
                    scatterDiv.innerHTML = '<p>Données insuffisantes pour générer le scatter plot.</p>';
                }
                return;
            }
            
            // Validation des tailles de tableaux
            if (scatterData.x.length !== scatterData.y.length || 
                scatterData.x.length !== (scatterData.labels || []).length) {
                console.warn('Tailles de données incohérentes pour le scatter plot');
                return;
            }
            
            // Continuer avec la génération du graphique...
            const trace = {
                x: scatterData.x,
                y: scatterData.y,
                mode: 'markers',
                type: 'scatter',
                text: scatterData.labels || [],
                marker: {
                    color: (scatterData.similarities || []).map(sim => 
                        sim >= (scatterData.threshold || 0.6) ? 'red' : 'blue'
                    ),
                    size: 8
                }
            };
            
            Plotly.newPlot('scatter-plot', [trace], {
                title: scatterData.title || 'Scatter Plot',
                xaxis: { title: 't-SNE Component 1' },
                yaxis: { title: 't-SNE Component 2' }
            });
        }

        function updateHeatmap(selectedKey) {
            /**
             * Met à jour la heatmap avec validation des données
             */
            const embeddings = safeGetData(processedData, `files.${selectedKey}.embeddings`, {});
            const embeddingKey = getSelectedEmbeddingKey();
            
            if (!embeddingKey || !(embeddingKey in embeddings)) {
                const heatmapDiv = document.getElementById('heatmap');
                if (heatmapDiv) {
                    heatmapDiv.innerHTML = '<p>Aucune donnée de heatmap disponible pour cette configuration.</p>';
                }
                return;
            }
            
            const fileData = validateFileData(embeddings[embeddingKey]);
            
            if (!fileData.phrases || !fileData.similarities || 
                fileData.phrases.length === 0 || fileData.similarities.length === 0) {
                const heatmapDiv = document.getElementById('heatmap');
                if (heatmapDiv) {
                    heatmapDiv.innerHTML = '<p>Données insuffisantes pour générer la heatmap.</p>';
                }
                return;
            }
            
            // Validation des tailles
            if (fileData.phrases.length !== fileData.similarities.length) {
                console.warn('Nombre de phrases et de similarités incohérent');
                return;
            }
            
            // Continuer avec la génération de la heatmap...
            // ... code existant de génération de heatmap
        }

        function updateMetricsTable(selectedKey) {
            /**
             * Met à jour le tableau des métriques avec validation
             */
            const embeddings = safeGetData(processedData, `files.${selectedKey}.embeddings`, {});
            const embeddingKey = getSelectedEmbeddingKey();
            
            if (!embeddingKey || !(embeddingKey in embeddings)) {
                const tableBody = document.querySelector('#metrics-table tbody');
                if (tableBody) {
                    tableBody.innerHTML = '<tr><td colspan="2">Aucune métrique disponible</td></tr>';
                }
                return;
            }
            
            const fileData = validateFileData(embeddings[embeddingKey]);
            const metrics = fileData.metrics;
            
            const tableBody = document.querySelector('#metrics-table tbody');
            if (!tableBody) return;
            
            tableBody.innerHTML = '';
            
            // Métriques essentielles avec valeurs par défaut
            const essentialMetrics = [
                { key: 'silhouette_score', label: 'Silhouette Score', format: (v) => v.toFixed(3) },
                { key: 'mean_similarity', label: 'Mean Similarity', format: (v) => v.toFixed(3) },
                { key: 'n_clusters', label: 'Number of Clusters', format: (v) => v.toString() }
            ];
            
            essentialMetrics.forEach(({ key, label, format }) => {
                const value = metrics[key];
                const formattedValue = (value !== undefined && value !== null) ? format(value) : 'N/A';
                
                const row = document.createElement('tr');
                row.innerHTML = `<td>${label}</td><td>${formattedValue}</td>`;
                tableBody.appendChild(row);
            });
            
            // Ajouter les autres métriques disponibles
            Object.entries(metrics).forEach(([key, value]) => {
                if (!essentialMetrics.some(m => m.key === key)) {
                    const formattedValue = (typeof value === 'number') ? value.toFixed(3) : value.toString();
                    const row = document.createElement('tr');
                    row.innerHTML = `<td>${key}</td><td>${formattedValue}</td>`;
                    tableBody.appendChild(row);
                }
            });
        }

        // Modifier les event listeners pour utiliser les nouvelles fonctions
        document.addEventListener('DOMContentLoaded', function() {
            processedData = decodeData(b64Data);
            if (processedData) {
                initialize();
            }
            
            // Ajouter une validation globale des données au chargement
            if (!processedData || !processedData.files) {
                console.error('Données processées manquantes ou invalides');
                document.body.innerHTML = '<h1>Erreur: Données manquantes</h1><p>Les données n\'ont pas pu être chargées correctement.</p>';
                return;
            }
            
            // Continuer avec l'initialisation normale...
        });
        """
        minified_js = jsmin(js_content)

        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
    <style>{minified_css}</style>
</head>
<body>
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
            <div id="file-links-container" class="file-links-grid"></div>
            <div id="scatter-plot-container"></div>
        </div>
    </div>

    <script>{minified_js}</script>
</body>
</html>
"""
        output_path = os.path.join(output_dir, job["filename"])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated main page at: {output_path}")
