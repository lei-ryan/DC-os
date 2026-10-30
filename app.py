from __future__ import annotations

import csv
import io
import math
import os

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html

from thermalgraph import ResearchArchive


archive = ResearchArchive()
metadata = [archive.node_metadata(node, index) for index, node in enumerate(archive.nodes)]

app = Dash(__name__, title="DC ThermalGraph Risk OS", suppress_callback_exceptions=True)
server = app.server


@server.route("/health")
def health():
    return {"status": "ok", "nodes": 72, "frames": archive.frame_count}


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16, "color": "#667085"})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return figure


def field_figure(frame: int, field: str) -> go.Figure:
    temperatures = archive.current_temperatures(frame)
    if field == "risk":
        values = [archive.risk_summary(frame, index, 60)["probability"] * 100 for index in range(72)]
        title = "Future 60 min thermal-deviation probability"
        scale = [[0, "#2ca25f"], [0.5, "#f6d55c"], [1, "#d73027"]]
        minimum, maximum, suffix = 0, 100, "%"
    else:
        values = temperatures
        title = "Current temperature field"
        scale = "Turbo"
        minimum, maximum, suffix = 28, 43, " °C"

    figure = go.Figure()
    edge_colors = {
        "front_to_back": "rgba(46,196,182,.45)",
        "vertical": "rgba(128,90,213,.30)",
        "same_row_adjacent": "rgba(235,160,45,.25)",
        "cross_row_matched": "rgba(52,152,219,.25)",
    }
    for edge_type, color in edge_colors.items():
        x_values, y_values, z_values = [], [], []
        for edge in archive.topology["directed_edges"]:
            if edge["edge_type"] != edge_type:
                continue
            source = metadata[edge["source_index"]]
            target = metadata[edge["target_index"]]
            x_values += [source["x"], target["x"], None]
            y_values += [source["y"], target["y"], None]
            z_values += [source["z"], target["z"], None]
        figure.add_trace(go.Scatter3d(x=x_values, y=y_values, z=z_values, mode="lines",
                                      line={"color": color, "width": 2}, hoverinfo="skip",
                                      showlegend=False))
    figure.add_trace(go.Scatter3d(
        x=[item["x"] for item in metadata], y=[item["y"] for item in metadata], z=[item["z"] for item in metadata],
        mode="markers+text", text=[item["rack"] if item["side"] == "Front" and item["level"] == "Upper" else "" for item in metadata],
        textposition="top center", customdata=[[item["index"], item["id"], item["side"], item["level"]] for item in metadata],
        marker={"size": 8, "color": values, "colorscale": scale, "cmin": minimum, "cmax": maximum,
                "colorbar": {"title": title, "ticksuffix": suffix, "thickness": 14}, "line": {"color": "#ffffff", "width": 1}},
        hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]} · %{customdata[3]}<br>Value: %{marker.color:.2f}" + suffix + "<extra></extra>",
        showlegend=False,
    ))
    figure.update_layout(
        template="plotly_white", margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=590,
        scene={"xaxis": {"title": "Rack column", "showbackground": False},
               "yaxis": {"title": "Row and face", "showbackground": False},
               "zaxis": {"title": "Height", "showbackground": False},
               "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.05}}, "aspectmode": "manual",
               "aspectratio": {"x": 1.8, "y": 1.0, "z": 0.75}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", uirevision="field-camera",
    )
    return figure


def trajectory_figure(frame: int, node: int) -> go.Figure:
    offset = archive.offsets[frame]
    history_start = max(0, offset - 60)
    history = [archive.temperature[index][node] for index in range(history_start, offset + 1)]
    forecast = [archive.prediction[frame][horizon][node] for horizon in range(60)]
    summaries = [archive.risk_summary(frame, node, horizon) for horizon in range(1, 61)]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(-len(history) + 1, 1)), y=history, name="Observed", line={"color": "#111827", "width": 2}))
    figure.add_trace(go.Scatter(x=list(range(1, 61)), y=[item["upper"] for item in summaries], mode="lines",
                                line={"width": 0}, hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(x=list(range(1, 61)), y=[item["lower"] for item in summaries], mode="lines",
                                fill="tonexty", fillcolor="rgba(14,165,164,.16)", line={"width": 0}, name="90% empirical interval"))
    figure.add_trace(go.Scatter(x=list(range(1, 61)), y=forecast, name="Frozen forecast", line={"color": "#0e8f8f", "width": 3}))
    figure.add_vline(x=0, line_dash="dot", line_color="#98a2b3")
    figure.update_layout(template="plotly_white", height=360, margin={"l": 48, "r": 20, "t": 20, "b": 40},
                         xaxis_title="Minutes relative to issue time", yaxis_title="Temperature / °C",
                         legend={"orientation": "h", "y": 1.08}, hovermode="x unified")
    return figure


def risk_figure(frame: int, node: int) -> go.Figure:
    probabilities = [archive.risk_summary(frame, node, horizon)["probability"] * 100 for horizon in range(1, 61)]
    colors = ["#2ca25f" if value < 25 else "#f0b429" if value < 50 else "#d73027" for value in probabilities]
    figure = go.Figure(go.Bar(x=list(range(1, 61)), y=probabilities, marker_color=colors, name="Risk probability"))
    figure.add_hline(y=50, line_dash="dash", line_color="#c2410c", annotation_text="Notification threshold 50%")
    figure.update_layout(template="plotly_white", height=330, margin={"l": 45, "r": 20, "t": 20, "b": 40},
                         xaxis_title="Forecast horizon / min", yaxis_title="Probability / %", yaxis_range=[0, 100])
    return figure


def metric_card(label: str, value: str, note: str = ""):
    return html.Div([html.Small(label), html.Strong(value), html.Span(note)], className="metric-card")


def frame_controls():
    return html.Div([
        html.Div([html.Label("Historical replay"), dcc.Slider(0, archive.frame_count - 1, 1, value=60,
            marks={0: "09:00", 60: "10:00", 120: "11:00", 180: "12:00"}, id="frame-slider")], className="control-wide"),
        html.Div([html.Label("Selected monitoring node"), dcc.Dropdown(
            [{"label": node, "value": index} for index, node in enumerate(archive.nodes)], value=40, clearable=False, id="node-select")]),
    ], className="control-strip")


def spatial_tab():
    return dcc.Tab(label="01 Thermal Field Situation", value="spatial", children=[
        html.Div([
            html.Div([html.Div([html.H2("Data-center 3D spatial mapping"), html.P("R05 / R06 · 12 racks · 72 monitoring nodes")], className="section-heading"),
                      dcc.RadioItems([{"label": "Current temperature", "value": "temperature"}, {"label": "Future risk at t + 60 min", "value": "risk"}],
                                     value="temperature", inline=True, id="field-select", className="segmented"),
                      dcc.Graph(id="field-graph", config={"displaylogo": False})], className="panel main-panel"),
            html.Div(id="selection-panel", className="panel side-panel"),
        ], className="two-column"),
        html.Div([html.Div([html.H3("Observed history and frozen continuous trajectory"), dcc.Graph(id="trajectory-graph", config={"displaylogo": False})], className="panel"),
                  html.Div([html.H3("Future 60 min risk evolution"), dcc.Graph(id="risk-graph", config={"displaylogo": False})], className="panel")], className="two-column equal")
    ])


def risk_tab():
    return dcc.Tab(label="02 Risk Inference", value="risk", children=[
        html.Div([html.Div([html.H2("Rolling thermal-risk inference"), html.P("Empirical residual risk-confidence mapping; not a ground-truth anomaly label")], className="section-heading"),
                  html.Div(id="risk-summary", className="metrics-grid")], className="panel"),
        html.Div([html.Div([html.H3("Future 1–60 min risk probability"), dcc.Graph(id="risk-graph-copy", config={"displaylogo": False})], className="panel"),
                  html.Div([html.H3("Four independent observation references"), html.Div(id="evidence-cards", className="evidence-grid")], className="panel")], className="two-column equal"),
        html.Div([html.H3("Node-level spatial review"), html.Div(id="top-risk-table")], className="panel")
    ])


def action_tab():
    return dcc.Tab(label="03 Recommended Actions", value="actions", children=[
        html.Div([html.Div([html.H2("Physics-consistent inspection recommendation"),
                           html.P("Human-in-the-loop guidance linked to archived evidence. No automatic control command is issued.")], className="section-heading"),
                  html.Button("Export inspection record", id="export-button", className="primary-button"), dcc.Download(id="download-record")], className="panel action-header"),
        html.Div([html.Div(id="action-content", className="panel"),
                  html.Div([html.H3("Decision provenance"), html.Ul([
                      html.Li("The spatial node determines the inspection object."),
                      html.Li("The frozen trajectory and archived residuals determine risk confidence."),
                      html.Li("RC graph neighbours determine the spatial review scope."),
                      html.Li("A human operator verifies and records the final finding."),
                  ])], className="panel")], className="two-column equal")
    ])


app.layout = html.Div([
    html.Header([html.Div([html.Span("TG", className="logo"), html.Div([html.H1("DC ThermalGraph Risk OS"), html.P("Temperature-field forecasting · risk inference · spatial review")])]),
                 html.Div(id="clock-label", className="archive-label")], className="topbar"),
    html.Main([frame_controls(), dcc.Tabs(id="main-tabs", value="spatial", children=[spatial_tab(), risk_tab(), action_tab()])], className="workspace"),
    html.Footer("Dissertation research prototype · TDC2.0 archived demonstration · decision support only"),
])


@callback(
    Output("field-graph", "figure"), Output("trajectory-graph", "figure"), Output("risk-graph", "figure"),
    Output("risk-graph-copy", "figure"), Output("selection-panel", "children"), Output("risk-summary", "children"),
    Output("evidence-cards", "children"), Output("top-risk-table", "children"), Output("action-content", "children"),
    Output("clock-label", "children"),
    Input("frame-slider", "value"), Input("node-select", "value"), Input("field-select", "value"),
)
def update_dashboard(frame: int, node: int, field: str):
    current = archive.current_temperatures(frame)[node]
    summary = archive.risk_summary(frame, node, 60)
    meta = metadata[node]
    risk_percent = summary["probability"] * 100
    evidence = archive.observation_evidence(frame, node)
    evidence_labels = {
        "statistical": "Temporal-cycle residual",
        "graph": "Physical-topology support",
        "dynamic": "Multivariate dynamic regime",
        "sequence": "Multiscale temporal variation",
    }
    evidence_cards = [html.Div([html.Small(f"0{index}"), html.B(evidence_labels[key]), html.Strong(f"{evidence[key] * 100:.0f}%"),
                                     html.Div(html.Span(style={"width": f"{evidence[key] * 100:.0f}%"}), className="probability-bar")], className="evidence-card")
                      for index, key in enumerate(evidence_labels, 1)]
    ranked = sorted([(archive.risk_summary(frame, idx, 60)["probability"], idx) for idx in range(72)], reverse=True)[:8]
    table = html.Table([html.Thead(html.Tr([html.Th("Rank"), html.Th("Node"), html.Th("Current"), html.Th("t + 60 forecast"), html.Th("Probability")])),
                        html.Tbody([html.Tr([html.Td(rank), html.Td(archive.nodes[idx]),
                                            html.Td(f"{archive.current_temperatures(frame)[idx]:.2f} °C"),
                                            html.Td(f"{archive.forecast(frame, 60)[idx]:.2f} °C"), html.Td(f"{probability * 100:.1f}%")])
                                    for rank, (probability, idx) in enumerate(ranked, 1)])], className="data-table")
    selection = [html.Small("CURRENT SELECTION"), html.H2(meta["id"]), html.P(f"{meta['rack']} · {meta['side']} · {meta['level']}"),
                 metric_card("Current temperature", f"{current:.2f} °C"), metric_card("t + 60 min forecast", f"{summary['prediction']:.2f} °C"),
                 metric_card("Thermal-deviation probability", f"{risk_percent:.1f}%"),
                 metric_card("90% empirical interval", f"{summary['lower']:.2f}–{summary['upper']:.2f} °C", f"n = {summary['sample_count']} residual samples"),
                 html.Div("Probability is calculated from the frozen trajectory and archived forecast-residual distribution.", className="notice")]
    risk_metrics = [metric_card("Selected node", meta["id"], f"{meta['side']} · {meta['level']}"),
                    metric_card("Current observation", f"{current:.2f} °C"), metric_card("t + 60 risk", f"{risk_percent:.1f}%"),
                    metric_card("Notification state", "Notify" if risk_percent >= 50 else "Monitor", "50% probability threshold")]
    action = [html.Small("HUMAN IN THE LOOP"), html.H2(f"Inspect {meta['id']} temperature and airflow conditions"),
              html.Div([metric_card("Priority", "P1" if risk_percent >= 50 else "P2"), metric_card("Object", f"{meta['rack']} · {meta['side']}"),
                        metric_card("Archived issue time", archive.origins[frame]), metric_card("Risk confidence", f"{risk_percent:.1f}%")], className="metrics-grid"),
              html.H3("Recommended inspection sequence"), html.Ol([
                  html.Li("Verify sensor validity and timestamp consistency."),
                  html.Li("Compare same-height front and rear temperatures."),
                  html.Li("Inspect adjacent racks and candidate RC graph edges for coordinated deviation."),
                  html.Li("Check local airflow obstruction, fan operation, and cooling delivery before intervention."),
              ]), html.Div("The recommendation prioritizes physical verification; it does not identify a validated root cause or authorize control changes.", className="notice warning")]
    risk_plot = risk_figure(frame, node)
    return field_figure(frame, field), trajectory_figure(frame, node), risk_plot, risk_plot, selection, risk_metrics, evidence_cards, table, action, f"Archived replay · {archive.origins[frame]}"


@callback(Output("download-record", "data"), Input("export-button", "n_clicks"), State("frame-slider", "value"), State("node-select", "value"), prevent_initial_call=True)
def export_record(_: int, frame: int, node: int):
    meta = metadata[node]
    summary = archive.risk_summary(frame, node, 60)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["issue_time", "node", "rack", "side", "level", "current_c", "forecast_t60_c", "risk_probability", "recommended_action"])
    writer.writerow([archive.origins[frame], meta["id"], meta["rack"], meta["side"], meta["level"],
                     archive.current_temperatures(frame)[node], summary["prediction"], summary["probability"],
                     "Verify sensor, front-back temperature difference, adjacent racks, and local airflow conditions."])
    return dcc.send_string(stream.getvalue(), f"inspection-{meta['id']}-{archive.origins[frame].replace(':', '').replace(' ', '-')}.csv")


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
