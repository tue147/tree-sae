"""Interactive HTML visualisation of a Tree SAE subtree (Figs. 17-20).

``save_feature_tree`` writes a self-contained D3 page: every node shows the feature's firing rate and
its max-activating examples on hover/click.
"""

from __future__ import annotations

import json
from html import escape

from torch import Tensor
from transformer_lens import HookedTransformer

from ..models import TreeSAE
from .features import firing_rate, max_activating_examples


def subtree(sae: TreeSAE, root: int) -> dict[int, list[int]]:
    """``{feature: children}`` for ``root`` and all its descendants (breadth-first)."""
    tree: dict[int, list[int]] = {}
    queue = [root]
    while queue:
        feature = queue.pop(0)
        tree[feature] = sae.children_of(feature)
        queue.extend(tree[feature])
    return tree


def save_feature_tree(child_features_tree, feature_data, filename="feature_tree.html"):
    """
    Save the feature tree visualization to an HTML file with hover tooltips
    and a scrollable, clickable details panel that pins node information.

    Parameters:
    child_features_tree - dictionary of parent-child relationships
    feature_data - dictionary containing feature details (sparsity, top activations)
    filename - name of the HTML file to save
    """

    # Generate the tree structure in JSON format
    def build_tree(parent_id, tree_data):
        children = []
        for child_id in tree_data.get(parent_id, []):
            node = {
                "name": f"Feature {child_id}",
                "feature_id": child_id,
                "sparsity": feature_data[child_id]["sparsity"],
                "top_activations": feature_data[child_id]["top_activations"],
                "children": build_tree(child_id, tree_data),
            }
            children.append(node)
        return children

    # Find the root (node with no parent)
    root_id = None
    all_children = set()
    for children in child_features_tree.values():
        all_children.update(children)

    for parent in child_features_tree:
        if parent not in all_children:
            root_id = parent
            break

    if root_id is None:
        root_id = list(child_features_tree.keys())[0]

    tree_structure = {
        "name": f"Feature {root_id}",
        "feature_id": root_id,
        "sparsity": feature_data[root_id]["sparsity"],
        "top_activations": feature_data[root_id]["top_activations"],
        "children": build_tree(root_id, child_features_tree),
    }

    # HTML and JavaScript for visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Feature Tree Visualization</title>
        <style>
            :root {{
                --panel-width: 420px;
                --panel-max-height: 80vh;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f5f5f5;
                color: #333;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 16px;
            }}

            h1 {{
                color: #333;
                text-align: center;
                margin: 12px 0 20px 0;
            }}

            .layout {{
                display: flex;
                gap: 16px;
                align-items: flex-start;
            }}

            #tree-wrapper {{
                flex: 1 1 auto;
                min-width: 0;
            }}

            #tree-container {{
                overflow: auto;
                border: 1px solid #ddd;
                padding: 10px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}

            /* Details panel */
            #details-panel {{
                width: var(--panel-width);
                max-width: 100%;
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                padding: 12px;
                position: sticky;
                top: 12px;
            }}

            #details-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                margin-bottom: 8px;
            }}

            #details-title {{
                font-size: 18px;
                font-weight: bold;
                margin: 0;
            }}

            #clear-btn {{
                background: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 6px 10px;
                cursor: pointer;
                font-size: 12px;
            }}
            #clear-btn:hover {{
                background: #eee;
            }}

            #details-content {{
                max-height: var(--panel-max-height);
                overflow-y: auto;
                padding-right: 4px;
            }}

            .kv {{
                margin: 6px 0 10px 0;
                font-size: 14px;
            }}

            .section-title {{
                font-weight: bold;
                margin: 8px 0 6px 0;
            }}

            .data-table,
            .tooltip-table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}

            .data-table th, .data-table td,
            .tooltip-table th, .tooltip-table td {{
                padding: 6px;
                border: 1px solid #ddd;
                text-align: left;
                vertical-align: top;
                font-size: 12px;
                background: #fff;
            }}

            .data-table th, .tooltip-table th {{
                background-color: #fafafa;
                font-weight: 600;
            }}

            /* Make long token sequences wrap gracefully */
            .data-table td:nth-child(2), .data-table td:nth-child(3),
            .tooltip-table td:nth-child(2), .tooltip-table td:nth-child(3) {{
                word-break: break-word;
                white-space: normal;
            }}

            .node circle {{
                fill: #fff;
                stroke: steelblue;
                stroke-width: 3px;
                transition: fill 0.2s ease, stroke 0.2s ease;
            }}
            
            .node text {{
                font: 12px sans-serif;
            }}
            
            .link {{
                fill: none;
                stroke: #ccc;
                stroke-width: 2px;
            }}

            .node.selected circle {{
                fill: #ffe082;
                stroke: #f57c00;
            }}
            
            .highlight {{
                background-color: #2e7d32;
                color: #fff;
                font-weight: bold;
                padding: 2px 1px;
                border-radius: 3px;
            }}
            
            #tooltip {{
                position: absolute;
                padding: 10px;
                background: rgba(0, 0, 0, 0.85);
                color: #fff;
                border-radius: 6px;
                pointer-events: none;
                font: 12px sans-serif;
                max-width: 420px;
                max-height: 70vh;
                overflow-y: auto;
                z-index: 9999;
                box-shadow: 0 4px 16px rgba(0,0,0,0.25);
            }}

            #tooltip .tooltip-table th, 
            #tooltip .tooltip-table td {{
                border-color: #555;
                background: transparent;
            }}

            #tooltip .tooltip-table th {{
                background-color: #444;
            }}

            /* Responsive layout: stack panel below tree on narrow screens */
            @media (max-width: 900px) {{
                .layout {{
                    flex-direction: column;
                }}
                #details-panel {{
                    width: 100%;
                    position: relative;
                    top: auto;
                }}
                #details-content {{
                    max-height: 40vh;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feature Tree Visualization</h1>
            <div class="layout">
                <div id="tree-wrapper">
                    <div id="tree-container"></div>
                </div>
                <aside id="details-panel" aria-live="polite">
                    <div id="details-header">
                        <h2 id="details-title">Node Details</h2>
                        <button id="clear-btn" type="button" title="Clear pinned details">Clear</button>
                    </div>
                    <div id="details-content">
                        <div class="kv">Hover a node to preview details. Click a node to pin its details here.</div>
                    </div>
                </aside>
            </div>
            <div id="tooltip" style="opacity:0"></div>
        </div>
        
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
            // Tree data
            const treeData = {json.dumps(tree_structure)};

            // Set up the tree
            const margin = {{top: 20, right: 120, bottom: 20, left: 120}};
            const width = 1200 - margin.right - margin.left;
            const height = 800 - margin.top - margin.bottom;
            
            const svg = d3.select("#tree-container").append("svg")
                .attr("width", width + margin.right + margin.left)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            
            // Create the tree layout
            const treemap = d3.tree().size([height, width]);
            
            // Convert the data to a hierarchy
            const root = d3.hierarchy(treeData);
            
            // Map the node data to the tree layout
            treemap(root);
            
            // Add links between nodes
            svg.selectAll(".link")
                .data(root.links())
                .enter().append("path")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal()
                    .x(function(d) {{ return d.y; }})
                    .y(function(d) {{ return d.x; }}));
            
            // Add each node
            const node = svg.selectAll(".node")
                .data(root.descendants())
                .enter().append("g")
                .attr("class", "node")
                .attr("transform", function(d) {{ 
                    return "translate(" + d.y + "," + d.x + ")";
                }})
                .on("mouseover", function(event, d) {{
                    showTooltip(event, d.data);
                }})
                .on("mouseout", function() {{
                    hideTooltip();
                }})
                .on("click", function(event, d) {{
                    // Pin details on click
                    event.stopPropagation();
                    pinDetails(d3.select(this), d.data);
                }});
            
            // Add circles to nodes
            node.append("circle")
                .attr("r", 10);
            
            // Add labels
            node.append("text")
                .attr("dy", ".35em")
                .attr("x", function(d) {{ return d.children ? -13 : 13; }})
                .attr("text-anchor", function(d) {{ return d.children ? "end" : "start"; }})
                .text(function(d) {{ return d.data.name; }});
            
            // Tooltip functions
            function showTooltip(event, data) {{
                const tooltip = d3.select("#tooltip");
                
                let html = "<strong>" + data.name + "</strong><br>";
                html += "Sparsity: " + Number(data.sparsity).toFixed(7) + "<br><br>";
                html += "<strong>Top Activations:</strong><br>";
                
                html += '<table class="tooltip-table">';
                html += '<tr><th style="width:80px">Activation</th><th>Sequence</th></tr>';
                
                (data.top_activations || []).forEach(function(act) {{
                    html += '<tr><td>' + Number(act.activation).toFixed(3) + '</td><td>' + 
                            act.sequence + '</td></tr>';
                }});
                
                html += '</table>';
                
                tooltip.html(html)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px")
                    .style("opacity", 1);
            }}
            
            function hideTooltip() {{
                d3.select("#tooltip")
                    .style("opacity", 0);
            }}

            // Details panel rendering
            const detailsContent = document.getElementById("details-content");
            const detailsTitle = document.getElementById("details-title");
            const clearBtn = document.getElementById("clear-btn");

            let selectedNodeSel = null; // d3 selection for styling

            function renderDetails(data) {{
                detailsTitle.textContent = data.name || "Node Details";

                let html = "";
                html += '<div class="kv"><strong>Sparsity:</strong> ' + Number(data.sparsity).toFixed(7) + "</div>";
                html += '<div class="section-title">Top Activations</div>';
                html += '<table class="data-table">';
                html += '<tr><th style="width:100px">Activation</th><th>Sequence</th></tr>';

                (data.top_activations || []).forEach(function(act) {{
                    html += '<tr>';
                    html += '<td>' + Number(act.activation).toFixed(3) + '</td>';
                    html += '<td>' + act.sequence + '</td>';
                    html += '</tr>';
                }});

                html += '</table>';
                detailsContent.innerHTML = html;

                // Reset scroll to top when a new node is pinned
                detailsContent.scrollTop = 0;
            }}

            function clearDetails() {{
                detailsTitle.textContent = "Node Details";
                detailsContent.innerHTML = '<div class="kv">Hover a node to preview details. Click a node to pin its details here.</div>';
            }}

            clearBtn.addEventListener("click", function() {{
                clearDetails();
                if (selectedNodeSel) {{
                    selectedNodeSel.classed("selected", false);
                    selectedNodeSel = null;
                }}
            }});

            function pinDetails(nodeSelection, data) {{
                // Remove previous selection style
                if (selectedNodeSel && selectedNodeSel.node() !== nodeSelection.node()) {{
                    selectedNodeSel.classed("selected", false);
                }}
                // Apply new selection style
                selectedNodeSel = nodeSelection;
                selectedNodeSel.classed("selected", true);

                renderDetails(data);
            }}

            // Optional: Pin the root by default
            // pinDetails(node.filter(d => d.depth === 0), root.data);

            // Hide tooltip if user clicks anywhere outside nodes
            document.addEventListener("click", function() {{
                hideTooltip();
            }});
        </script>
    </body>
    </html>
    """

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)


def format_sequence_with_highlight(str_toks, highlight_pos, max_chars=300):
    # Same logic but with truncation for large sequences to reduce HTML size
    formatted_tokens = []
    total_len = 0
    for i, token in enumerate(str_toks):
        cleaned_token = token.replace("�", "").replace("\n", "↵")
        cleaned_token = escape(cleaned_token)
        if i == highlight_pos:
            cleaned_token = f'<span class="highlight">{cleaned_token}</span>'
        new_len = total_len + len(cleaned_token)
        if max_chars is not None and new_len > max_chars:
            formatted_tokens.append(" …")
            break
        formatted_tokens.append(cleaned_token)
        total_len = new_len
    return "".join(formatted_tokens)


def build_feature_data(
    tree: dict[int, list[int]],
    model: HookedTransformer,
    indices: Tensor,
    values: Tensor,
    tokens: Tensor,
    k: int = 10,
    buffer: int = 15,
    max_seq_chars: int = 300,
) -> dict[int, dict]:
    """Firing rate and top-``k`` examples of every feature in ``tree``.

    ``indices``/``values`` are ``(n_seqs, seq_len, k)`` sparse activations of ``tokens``.
    """
    features = set(tree) | {c for children in tree.values() for c in children}
    data = {}
    for feature in sorted(features):
        examples = max_activating_examples(model, feature, indices, values, tokens, k=k, buffer=buffer)
        data[feature] = {
            "sparsity": firing_rate(feature, indices, values),
            "top_activations": [
                {"activation": float(act), "sequence": format_sequence_with_highlight(toks, pos, max_seq_chars)}
                for act, toks, pos in examples
            ],
        }
    return data
