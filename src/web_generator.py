import json
import os
from typing import Any, Dict


def generate_main_page(processed_data: Dict[str, Any], output_dir: str):
    """
    Génère la page web principale interactive pour la visualisation des heatmaps.
    """

    # Sérialiser les données pour les injecter dans le JavaScript
    data_json = json.dumps(processed_data, indent=4)

    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse d'Embeddings</title>
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
        .controls, .metrics, .visualization {{
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
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 15px;
        }}
        .control-group label {{
            font-weight: bold;
            min-width: 150px;
        }}
        .slider-container {{
            flex-grow: 1;
            display: flex;
            align-items: center;
            gap: 15px;
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
        #file-name, #embedding-name {{
            font-weight: bold;
            color: #2980b9;
            width: 350px; /* Largeur fixe */
            min-width: 350px; /* Assure que la largeur ne diminue pas */
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
            flex: 0 0 100px; /* Largeur réduite */
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
            min-height: 60vh; /* Hauteur minimale augmentée */
        }}
        .heatmap-content span {{
            padding: 2px 1px;
            border-radius: 3px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Analyse Interactive des Embeddings</h1>

        <!-- Zone de contrôles -->
        <div class="controls">
            <h2>Contrôles</h2>
            <div class="control-group">
                <label for="file-slider">Fichier Markdown :</label>
                <div class="slider-container">
                    <input type="range" id="file-slider" min="0" max="0" value="0">
                    <span id="file-name"></span>
                </div>
            </div>
            <div class="control-group">
                <label for="embedding-slider">Paramétrage d'Embedding :</label>
                <div class="slider-container">
                    <input type="range" id="embedding-slider" min="0" max="0" value="0">
                    <span id="embedding-name"></span>
                </div>
            </div>
        </div>

        <!-- Zone de métriques -->
        <div class="metrics">
            <h2>Métriques</h2>
            <div id="metrics-grid" class="metrics-grid"></div>
        </div>

        <!-- Zone de visualisation -->
        <div class="visualization">
            <h2>Heatmap Textuelle</h2>
            <div id="heatmap-container" class="heatmap-content"></div>
        </div>
    </div>

    <script>
        const processedData = {json.dumps(processed_data, indent=4)};

        const fileSlider = document.getElementById('file-slider');
        const embeddingSlider = document.getElementById('embedding-slider');
        const fileNameSpan = document.getElementById('file-name');
        const embeddingNameSpan = document.getElementById('embedding-name');
        const metricsGrid = document.getElementById('metrics-grid');
        const heatmapContainer = document.getElementById('heatmap-container');

        let fileKeys = [];
        let embeddingKeys = [];

        function initialize() {{
            fileKeys = Object.keys(processedData.files);
            if (fileKeys.length === 0) return;

            fileSlider.max = fileKeys.length - 1;
            fileSlider.addEventListener('input', updateInterface);
            embeddingSlider.addEventListener('input', updateInterface);

            updateInterface();
        }}

        function updateInterface() {{
            const fileIndex = parseInt(fileSlider.value, 10);
            const fileKey = fileKeys[fileIndex];
            const fileData = processedData.files[fileKey];
            
            fileNameSpan.textContent = fileKey;

            embeddingKeys = Object.keys(fileData.embeddings);
            embeddingSlider.max = embeddingKeys.length - 1;
            
            const embeddingIndex = parseInt(embeddingSlider.value, 10);
            if (embeddingIndex >= embeddingKeys.length) {{
                embeddingSlider.value = embeddingKeys.length - 1;
                updateInterface(); // Re-trigger update
                return;
            }}
            const embeddingKey = embeddingKeys[embeddingIndex];
            const embeddingData = fileData.embeddings[embeddingKey];

            embeddingNameSpan.textContent = embeddingKey;

            updateMetrics(embeddingData.metrics);
            updateHeatmap(fileData.phrases, embeddingData.similarities, embeddingData.themes);
        }}

        function updateMetrics(metrics) {{
            metricsGrid.innerHTML = '';
            for (const [key, value] of Object.entries(metrics)) {{
                const metricItem = document.createElement('div');
                metricItem.className = 'metric-item';
                
                const valueSpan = document.createElement('div');
                valueSpan.className = 'value';
                valueSpan.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                
                const labelSpan = document.createElement('div');
                labelSpan.className = 'label';
                labelSpan.textContent = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());

                metricItem.appendChild(valueSpan);
                metricItem.appendChild(labelSpan);
                metricsGrid.appendChild(metricItem);
            }}
        }}

        function updateHeatmap(phrases, similarities, themes) {{
            heatmapContainer.innerHTML = `<h3>Thèmes: ${{themes.join(', ')}}</h3>`;
            const content = document.createElement('p');

            const cmap = (score) => {{
                const colors = [
                    {{ r: 43, g: 131, b: 186 }}, // #2b83ba
                    {{ r: 171, g: 221, b: 164 }}, // #abdda4
                    {{ r: 255, g: 255, b: 191 }}, // #ffffbf
                    {{ r: 253, g: 174, b: 97 }},  // #fdae61
                    {{ r: 215, g: 25, b: 28 }}    // #d7191c
                ];
                
                const i = Math.min(Math.floor(score * (colors.length - 1)), colors.length - 2);
                const t = (score * (colors.length - 1)) % 1;
                
                const r = Math.round(colors[i].r * (1 - t) + colors[i+1].r * t);
                const g = Math.round(colors[i].g * (1 - t) + colors[i+1].g * t);
                const b = Math.round(colors[i].b * (1 - t) + colors[i+1].b * t);
                
                return `rgb(${{r}},${{g}},${{b}})`;
            }};

            phrases.forEach((phrase, index) => {{
                const score = similarities[index];
                const color = cmap(score);
                
                const span = document.createElement('span');
                span.style.backgroundColor = color;
                span.textContent = phrase + '. ';
                span.title = `Similarité: ${{score.toFixed(3)}}`;
                
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


def generate_model_page(model: Dict[str, Any], output_dir: str):
    """
    Génère une page HTML individuelle pour un modèle (potentiellement obsolète avec la nouvelle interface).
    Cette fonction est conservée pour la compatibilité mais pourrait être retirée.
    """

    model_name = model.get("name", "Unknown Model")
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Résultats pour {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .chart {{ margin-bottom: 30px; }}
        img {{ max-width: 100%; height: auto; }}
        a {{ color: #3498db; }}
    </style>
</head>
<body>
    <h1>Résultats pour le modèle : {model_name}</h1>
    <p><a href="index.html">Retour à la nouvelle page principale interactive</a></p>
    
    <h2>Métriques d'évaluation</h2>
    <ul>
"""
    for key, value in model.items():
        if key not in ["name", "type", "files", "embeddings_data"]:
            html_content += f"<li><strong>{key.replace('_', ' ').capitalize()}:</strong> {value:.4f}</li>"
    html_content += "</ul>"

    html_content += "<h2>Visualisations Graphiques</h2>"
    for file_info in model.get("files", []):
        if "path" in file_info and file_info["path"].endswith(".png"):
            file_path = os.path.basename(file_info["path"])
            html_content += f"""
            <div class="chart">
                <h3>{file_info.get("type", "Graphique")}</h3>
                <img src="{file_path}" alt="{file_info.get("type", "Graphique")}">
            </div>
            """

    html_content += """
</body>
</html>
"""

    output_path = os.path.join(output_dir, f"{model_name}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated legacy model page at: {output_path}")
