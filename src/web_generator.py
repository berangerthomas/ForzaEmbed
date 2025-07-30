import json
import os
from typing import Any, Dict


def generate_main_page(processed_data: Dict[str, Any], output_dir: str):
    """
    Generates the main interactive web page for heatmap visualization.
    """

    # Serialize the data to inject into JavaScript
    data_json = json.dumps(processed_data, indent=4)

    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f9;
            color: #333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            width: 100%;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        .controls {{
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
        }}
        .metrics, .visualization {{
            border: 1px solid #d1d9e0;
            border-radius: 8px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        .controls h2, .metrics h2, .visualization h2 {{
            margin-top: 0;
            color: #34495e;
            border-bottom: 2px solid #e0e6ed;
            padding-bottom: 10px;
        }}
        .control-group {{
            margin-bottom: 0;
            overflow: hidden;
        }}
        .label-group {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
        }}
        .control-group label {{
            font-weight: bold;
            font-size: 0.9em;
            display: flex;
            align-items: center;
        }}
        .label-text {{
            white-space: nowrap;
            margin-right: 0.5em;
        }}
        .slider-container {{
            flex-grow: 1;
            display: flex;
            align-items: center;
        }}
        input[type="range"] {{
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
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #3498db;
            cursor: pointer;
            border-radius: 50%;
            border: 2px solid #fff;
        }}
        input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            background: #3498db;
            cursor: pointer;
            border-radius: 50%;
            border: 2px solid #fff;
        }}
        .label-value {{
            font-weight: bold;
            color: #2980b9;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: right;
            flex-grow: 1;
            min-width: 0;
        }}
        .metrics {{
            flex-shrink: 0;
        }}
        .metrics-grid {{
            display: flex;
            flex-direction: row;
            gap: 15px;
            overflow-x: auto;
            padding-bottom: 10px;
        }}
        .metric-item {{
            flex: 0 0 100px;
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-item .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-item .label {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .visualization {{
            overflow-y: auto;
            line-height: 1.8;
            min-height: 60vh;
        }}
        .heatmap-content span {{
            padding: 2px 1px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.85em;
            line-height: 1.2;
        }}
        .links {{
            border: 1px solid #d1d9e0;
            border-radius: 8px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-top: 20px;
        }}
        .links h2 {{
            margin-top: 0;
            color: #34495e;
            border-bottom: 2px solid #e0e6ed;
            padding-bottom: 10px;
        }}
        #file-links-container a {{
            display: block;
            margin-bottom: 10px;
            color: #3498db;
            text-decoration: none;
        }}
        #file-links-container a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Embeddings Analysis</h1>

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

        <!-- Links area -->
        <div class="links">
            <h2>Generated Files</h2>
            <div id="file-links-container"></div>
        </div>
    </div>

    <script>
        const processedData = {data_json};

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
        const fileLinksContainer = document.getElementById('file-links-container');

        let fileKeys = [];
        let allEmbeddingKeys = [];
        let filteredEmbeddingKeys = [];
        const params = {{
            model: [],
            cs: [],
            co: [],
            t: [],
            s: [],
            m: []
        }};

        const createCmap = (colorSet) => (score) => {{
            if (typeof score !== 'number' || isNaN(score)) score = 0.5;
            const i = Math.min(Math.floor(score * (colorSet.length - 1)), colorSet.length - 2);
            const t = (score * (colorSet.length - 1)) % 1;
            const r = Math.round(colorSet[i].r * (1 - t) + colorSet[i+1].r * t);
            const g = Math.round(colorSet[i].g * (1 - t) + colorSet[i+1].g * t);
            const b = Math.round(colorSet[i].b * (1 - t) + colorSet[i+1].b * t);
            return {{
                rgb: `rgb(${{r}},${{g}},${{b}})`,
                isDark: (r * 0.299 + g * 0.587 + b * 0.114) < 128
            }};
        }};

        const cmap_metrics = createCmap([
            {{ r: 215, g: 25, b: 28 }},    // Bad
            {{ r: 253, g: 174, b: 97 }},
            {{ r: 255, g: 255, b: 191 }}, // Neutral
            {{ r: 171, g: 221, b: 164 }},
            {{ r: 43, g: 131, b: 186 }}  // Good
        ]);

        const cmap_heatmap = createCmap([
            {{ r: 43, g: 131, b: 186 }}, // Low similarity
            {{ r: 171, g: 221, b: 164 }},
            {{ r: 255, g: 255, b: 191 }}, // Neutral
            {{ r: 253, g: 174, b: 97 }},
            {{ r: 215, g: 25, b: 28 }}    // High similarity
        ]);

        function parseEmbeddingKey(key) {{
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
            
            return {{ model, cs, co, t, s, m }};
        }}

        function populateFilters() {{
            const firstFileKey = fileKeys[0];
            if (!firstFileKey) return;

            const paramSets = {{
                model: new Set(),
                cs: new Set(),
                co: new Set(),
                t: new Set(),
                s: new Set(),
                m: new Set()
            }};

            allEmbeddingKeys = Object.keys(processedData.files[firstFileKey].embeddings);
            allEmbeddingKeys.forEach(key => {{
                const p = parseEmbeddingKey(key);
                paramSets.model.add(p.model);
                paramSets.cs.add(p.cs);
                paramSets.co.add(p.co);
                paramSets.t.add(p.t);
                paramSets.s.add(p.s);
                paramSets.m.add(p.m);
            }});

            params.model = [...paramSets.model].sort();
            params.cs = [...paramSets.cs].sort((a, b) => a - b);
            params.co = [...paramSets.co].sort((a, b) => a - b);
            params.t = [...paramSets.t].sort();
            params.s = [...paramSets.s].sort();
            params.m = [...paramSets.m].sort();

            modelSlider.max = params.model.length - 1;
            chunkSizeSlider.max = params.cs.length - 1;
            chunkOverlapSlider.max = params.co.length - 1;
            themeSlider.max = params.t.length - 1;
            chunkingStrategySlider.max = params.s.length - 1;
            similarityMetricSlider.max = params.m.length - 1;
        }}

        function initialize() {{
            fileKeys = Object.keys(processedData.files || {{}});
            if (fileKeys.length === 0) {{
                console.error("No files found in processed data.");
                return;
            }}

            fileSlider.max = fileKeys.length - 1;
            populateFilters();

            fileSlider.addEventListener('input', updateInterface);
            modelSlider.addEventListener('input', updateInterface);
            chunkSizeSlider.addEventListener('input', updateInterface);
            chunkOverlapSlider.addEventListener('input', updateInterface);
            themeSlider.addEventListener('input', updateInterface);
            chunkingStrategySlider.addEventListener('input', updateInterface);
            similarityMetricSlider.addEventListener('input', updateInterface);

            updateInterface();
        }}

        function filterEmbeddings() {{
            const selectedModel = params.model[modelSlider.value];
            const selectedCS = params.cs[chunkSizeSlider.value];
            const selectedCO = params.co[chunkOverlapSlider.value];
            const selectedT = params.t[themeSlider.value];
            const selectedS = params.s[chunkingStrategySlider.value];
            const selectedM = params.m[similarityMetricSlider.value];

            modelNameSpan.textContent = selectedModel;
            chunkSizeValueSpan.textContent = selectedCS;
            chunkOverlapValueSpan.textContent = selectedCO;
            themeNameSpan.textContent = selectedT;
            chunkingStrategyNameSpan.textContent = selectedS;
            similarityMetricNameSpan.textContent = selectedM;

            filteredEmbeddingKeys = allEmbeddingKeys.filter(key => {{
                const p = parseEmbeddingKey(key);
                return p.model === selectedModel &&
                       p.cs === selectedCS &&
                       p.co === selectedCO &&
                       p.t === selectedT &&
                       p.s === selectedS &&
                       p.m === selectedM;
            }});
        }}

        function updateInterface() {{
            const fileIndex = parseInt(fileSlider.value, 10);
            const fileKey = fileKeys[fileIndex];
            const fileData = processedData.files[fileKey];
            
            if (!fileData) {{
                fileNameSpan.textContent = 'No data for this file.';
                metricsGrid.innerHTML = '';
                heatmapContainer.innerHTML = '';
                fileLinksContainer.innerHTML = '';
                return;
            }}

            fileNameSpan.textContent = fileKey;
            filterEmbeddings();

            if (filteredEmbeddingKeys.length === 0) {{
                metricsGrid.innerHTML = 'No data for this selection.';
                heatmapContainer.innerHTML = '';
                fileLinksContainer.innerHTML = '';
                return;
            }}

            // For simplicity, we display the first result of the filtered list.
            // A slider for filtered results could be added here.
            const embeddingKey = filteredEmbeddingKeys[0];
            const embeddingData = fileData.embeddings[embeddingKey];

            if (!embeddingData) {{
                metricsGrid.innerHTML = 'Error: Data not found.';
                heatmapContainer.innerHTML = '';
                fileLinksContainer.innerHTML = '';
                return;
            }}

            updateMetrics(embeddingData.metrics, fileKey);
            updateHeatmap(embeddingData.phrases, embeddingData.similarities);
            updateFileLinks(embeddingKey, fileKey);
        }}

        function updateFileLinks(embeddingKey, fileKey) {{
            fileLinksContainer.innerHTML = '';

            const filteredFileName = `${{embeddingKey}}_${{fileKey}}_filtered.md`;
            const explanatoryFileName = `${{embeddingKey}}_${{fileKey}}_explanatory.md`;

            const filteredLink = document.createElement('a');
            filteredLink.href = filteredFileName;
            filteredLink.textContent = `Filtered Report (filtered.md)`;
            filteredLink.title = filteredFileName;

            const explanatoryLink = document.createElement('a');
            explanatoryLink.href = explanatoryFileName;
            explanatoryLink.textContent = `Explanatory Report (explanatory.md)`;
            explanatoryLink.title = explanatoryFileName;

            fileLinksContainer.appendChild(filteredLink);
            fileLinksContainer.appendChild(explanatoryLink);
        }}

        function updateMetrics(metrics, fileKey) {{
            metricsGrid.innerHTML = '';
            if (!metrics) return;

            const lowerIsBetter = ["separation", "davies_bouldin", "processing_time"];
            
            const normalized = {{}};
            for (const key of Object.keys(metrics)) {{
                const allValues = filteredEmbeddingKeys
                    .map(ek => processedData.files[fileKey]?.embeddings[ek]?.metrics?.[key])
                    .filter(v => typeof v === 'number' && isFinite(v));
                
                if (allValues.length === 0) {{
                    normalized[key] = 0.5;
                    continue;
                }}

                const min = Math.min(...allValues);
                const max = Math.max(...allValues);
                let score = (metrics[key] - min) / (max - min || 1);
                if (lowerIsBetter.includes(key)) {{
                    score = 1 - score;
                }}
                normalized[key] = isNaN(score) ? 0.5 : score;
            }}

            for (const [key, value] of Object.entries(metrics)) {{
                const metricItem = document.createElement('div');
                metricItem.className = 'metric-item';
                
                const normValue = normalized[key];
                const colorInfo = cmap_metrics(normValue);
                metricItem.style.backgroundColor = colorInfo.rgb;
                
                const valueSpan = document.createElement('div');
                valueSpan.className = 'value';
                valueSpan.style.color = colorInfo.isDark ? '#fff' : '#2c3e50';
                valueSpan.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                
                const labelSpan = document.createElement('div');
                labelSpan.className = 'label';
                labelSpan.style.color = colorInfo.isDark ? '#ecf0f1' : '#7f8c8d';
                labelSpan.textContent = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());

                metricItem.appendChild(valueSpan);
                metricItem.appendChild(labelSpan);
                metricsGrid.appendChild(metricItem);
            }}
        }}

        function updateHeatmap(phrases, similarities) {{
            heatmapContainer.innerHTML = '';
            if (!phrases || !similarities) return;

            const content = document.createElement('p');

            phrases.forEach((phrase, index) => {{
                const score = similarities[index] || 0;
                const colorInfo = cmap_heatmap(score);
                
                const span = document.createElement('span');
                span.style.backgroundColor = colorInfo.rgb;
                span.style.color = colorInfo.isDark ? '#fff' : '#333';
                span.textContent = phrase + '. ';
                span.title = `Similarity: ${{score.toFixed(3)}}`;
                
                content.appendChild(span);
            }});
            heatmapContainer.appendChild(content);
        }}

        document.addEventListener('DOMContentLoaded', initialize);
    </script>
</body>
</html>
"""

    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated main page at: {output_path}")
