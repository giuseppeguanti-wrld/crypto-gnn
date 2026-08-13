"""Interactive graph explorer for the crypto-gnn correlation study (Sprint 6).

Reads only artifacts already produced by scripts/01-08 -- it never recomputes
the pipeline. The one exception is the threshold: comparing a correlation
matrix against tau is an instant NumPy operation, not a pipeline step.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cryptognn import artifacts, events
from cryptognn.config import Config, load_config
from cryptognn.graph.build import apply_threshold, edge_density, mantegna_weights
from cryptognn.graph.threshold import TauCalibration
from cryptognn.paths import DATA_PROCESSED, DEFAULT_CONFIG, DEFAULT_EVENTS
from cryptognn.viz.figures import TOPOLOGY_TIMESERIES_PANELS
from cryptognn.viz.graphs import draw_snapshot, fixed_layout
from cryptognn.viz.style import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    REFERENCE_COLOR,
    STACK_PANEL_HEIGHT,
    apply_style,
)
from cryptognn.viz.topology import draw_heatmap, draw_metric_series, hierarchical_order

st.set_page_config(layout="wide", page_title="Grafo di correlazione cripto")


@st.cache_data
def get_config() -> Config:
    return load_config(DEFAULT_CONFIG)


@st.cache_data
def load_returns():
    return artifacts.load_returns()


@st.cache_data
def load_corr(window: int):
    return artifacts.load_corr(window)


@st.cache_data
def load_topology():
    return artifacts.load_topology()


@st.cache_data
def load_tau() -> TauCalibration:
    return artifacts.load_tau()


@st.cache_data
def load_events() -> list[events.Event]:
    """Crisis event markers from config/events.yaml, same file as fig_topology_timeseries.pdf.

    Not behind the MissingArtifactError guard below: this is a committed config
    file, not a gitignored pipeline output, so its absence is a broken checkout
    rather than a missing pipeline step.
    """
    return events.load_events(DEFAULT_EVENTS)


@st.cache_resource
def load_layout() -> dict:
    """Node positions on the mean full-period graph, matching fig_graph_snapshots.pdf.

    Computed once and reused unchanged across reruns -- a fresh spring layout
    every time the slider moves would make the graph appear to reshuffle
    instead of just gaining or losing edges.
    """
    config = get_config()
    w_full = artifacts.load_w_full()
    return fixed_layout(w_full.mean(axis=0), seed=config.seed, labels=config.data.symbols)


@st.cache_resource
def load_heatmap_order():
    """Clustering order on the full-period mean correlation, matching fig_correlation_heatmaps.pdf.

    Tied to config.graph.window (not the sidebar's selected window): the
    ordering is a readability aid shared across every static figure, not a
    data value that should change when the user explores a sensitivity
    window.
    """
    config = get_config()
    corr, _ = load_corr(config.graph.window)
    return hierarchical_order(corr.mean(axis=0))


def available_windows() -> list[int]:
    """Every rolling-correlation window with a corr_{w}.npy already on disk.

    Scans data/processed/ rather than hardcoding [30, 60, 90]: only window=60
    has been computed so far, and offering an uncomputed window in the
    selectbox would trade a clear MissingArtifactError for a confusing one
    raised deep inside a cached loader.
    """
    windows = []
    for path in DATA_PROCESSED.glob("corr_*.npy"):
        try:
            windows.append(int(path.stem.removeprefix("corr_")))
        except ValueError:
            continue
    return sorted(windows)


def render_sidebar(config: Config, tau: TauCalibration) -> dict:
    st.sidebar.header("Controlli")

    windows = available_windows()
    default_index = windows.index(config.graph.window) if config.graph.window in windows else 0
    window = st.sidebar.selectbox("Finestra T_w (giorni)", windows, index=default_index)

    _corr, corr_index = load_corr(window)
    dates = list(corr_index)
    st.session_state.setdefault("date", dates[-1])
    if st.session_state["date"] not in dates:
        # The stored date can fall outside a narrower window's range; without
        # this the widget below would raise instead of just resetting.
        st.session_state["date"] = dates[-1]
    date = st.sidebar.select_slider(
        "Data", options=dates, key="date", format_func=lambda ts: ts.strftime("%Y-%m-%d")
    )

    st.session_state.setdefault("threshold_choice", "Calibrata")
    threshold_choice = st.sidebar.radio(
        "Soglia", ["Calibrata", "FWER", "Manuale"], key="threshold_choice"
    )
    if threshold_choice == "Calibrata":
        tau_value = tau.tau
    elif threshold_choice == "FWER":
        tau_value = tau.tau_fwer
    else:
        # This widget only exists while "Manuale" is selected. Streamlit drops
        # a widget's own key-bound state once that widget goes a rerun without
        # being instantiated, so a `key=` alone would reset to tau.tau every
        # time the user switches away and back. Persisting through a plain,
        # always-alive session_state entry -- read as `value=`, written back
        # after the call -- survives the widget's absence instead.
        st.session_state.setdefault("tau_manual_value", float(tau.tau))
        tau_value = st.sidebar.slider(
            "τ manuale", 0.0, 0.9, st.session_state["tau_manual_value"], 0.01
        )
        st.session_state["tau_manual_value"] = tau_value
        st.sidebar.caption(f"Riferimento: τ calibrata = {tau.tau:.4f}")

    fixed = st.sidebar.checkbox("Layout fisso", value=True)

    compare = st.sidebar.checkbox("Confronta due date", value=False)
    second_date = None
    if compare:
        # Same reasoning as tau_manual_value: this slider only exists while
        # "Confronta due date" is on, so it needs its own persistent shadow key
        # rather than relying on a widget key that disappears with it.
        st.session_state.setdefault("second_date_value", dates[0])
        if st.session_state["second_date_value"] not in dates:
            st.session_state["second_date_value"] = dates[0]
        second_date = st.sidebar.select_slider(
            "Seconda data", options=dates, value=st.session_state["second_date_value"],
            format_func=lambda ts: ts.strftime("%Y-%m-%d"),
        )
        st.session_state["second_date_value"] = second_date

    return {
        "window": window, "date": date, "second_date": second_date,
        "tau_value": tau_value, "threshold_choice": threshold_choice,
        "fixed_layout": fixed, "compare": compare,
    }


def _weights_for(corr, idx, tau_value):
    weights = mantegna_weights(corr[idx])
    thresh = apply_threshold(corr[idx], weights, tau_value)
    return weights, thresh


def _density_at(corr, corr_index, tau_value, date):
    idx = corr_index.get_loc(date)
    _weights, thresh = _weights_for(corr, idx, tau_value)
    return float(edge_density(thresh))


def _value_and_delta(date, index, value_at):
    """A metric's value at `date` and its change from 30 days earlier, or None if that date predates the record."""
    current = value_at(date)
    prior_date = date - pd.Timedelta(days=30)
    prior = value_at(prior_date) if prior_date in index else None
    return current, (None if prior is None else current - prior)


def _layout_for(weights, config, fixed, cached_layout):
    """The mean-graph layout when 'fixed' is on, otherwise a fresh spring layout for this date.

    Turning the toggle off is meant to show *why* a fixed layout matters:
    nodes reshuffle between dates instead of just gaining or losing edges.
    """
    return cached_layout if fixed else fixed_layout(weights, seed=config.seed, labels=config.data.symbols)


def _figure_pdf_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    return buffer.getvalue()


def render_topology_strip(topology: pd.DataFrame, events_list: list, date: pd.Timestamp) -> None:
    """The four topology panels of fig_topology_timeseries.pdf, plus the selected date.

    Same panels, same draw_metric_series() calls, same event markers as the
    static figure -- TOPOLOGY_TIMESERIES_PANELS is the single shared list. The
    one addition is a solid axvline at `date`: a full line in REFERENCE_COLOR,
    deliberately distinct from the dashed EVENT_COLOR used for crisis markers,
    so the current selection is never mistaken for a documented event.
    """
    panels = TOPOLOGY_TIMESERIES_PANELS
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(FIGURE_WIDTH, STACK_PANEL_HEIGHT * len(panels)), sharex=True
    )
    for ax, (metrics, labels, ylabel) in zip(axes, panels, strict=True):
        draw_metric_series(
            ax, topology, metrics, events=events_list, labels=labels, ylabel=ylabel,
            label_events=ax is axes[0],
        )
        ax.axvline(date, color=REFERENCE_COLOR, linewidth=1.0)
    axes[-1].set_xlabel("Data")
    fig.align_ylabels(axes)
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    apply_style()
    config = get_config()

    try:
        returns = load_returns()
        topology = load_topology()
        tau = load_tau()
        layout = load_layout()
    except artifacts.MissingArtifactError as exc:
        st.error(f"Artefatto mancante: `{exc.path}`\n\nEseguire prima:\n```\n{exc.command}\n```")
        st.stop()
        return
    events_list = load_events()

    st.title("Grafo di correlazione cripto")
    st.caption(
        f"{len(returns)} osservazioni · {returns.shape[1]} asset · "
        f"τ calibrata = {tau.tau:.4f} · τ FWER = {tau.tau_fwer:.4f} · "
        f"{len(topology)} righe di metriche topologiche · layout su {len(layout)} nodi"
    )

    selection = render_sidebar(config, tau)
    corr, corr_index = load_corr(selection["window"])
    date, tau_value = selection["date"], selection["tau_value"]
    date_idx = corr_index.get_loc(date)
    weights_t, thresh_t = _weights_for(corr, date_idx, tau_value)
    pos_t = _layout_for(weights_t, config, selection["fixed_layout"], layout)

    density_now, density_delta = _value_and_delta(
        date, corr_index, lambda d: _density_at(corr, corr_index, tau_value, d)
    )
    rho_now, rho_delta = _value_and_delta(date, topology.index, lambda d: topology.loc[d, "mean_correlation"])
    lambda2_now, lambda2_delta = _value_and_delta(
        date, topology.index, lambda d: topology.loc[d, "algebraic_connectivity_combinatorial"]
    )
    mst_now, mst_delta = _value_and_delta(date, topology.index, lambda d: topology.loc[d, "mst_length"])
    eigs_now, eigs_delta = _value_and_delta(date, topology.index, lambda d: topology.loc[d, "eigs_outside_mp"])

    st.subheader(f"Istantanea: {date.strftime('%Y-%m-%d')}")
    st.caption(
        f"Finestra T_w = {selection['window']}g · Soglia = {tau_value:.4f} ({selection['threshold_choice']}) · "
        "Δ rispetto a 30 giorni prima"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Densità", f"{density_now:.3f}", None if density_delta is None else f"{density_delta:+.3f}")
    m2.metric("ρ̄ media", f"{rho_now:.3f}", None if rho_delta is None else f"{rho_delta:+.3f}")
    m3.metric("λ2 (Fiedler)", f"{lambda2_now:.4f}", None if lambda2_delta is None else f"{lambda2_delta:+.4f}")
    m4.metric("Lunghezza MST", f"{mst_now:.3f}", None if mst_delta is None else f"{mst_delta:+.3f}")
    m5.metric("Autoval. fuori MP", f"{int(eigs_now)}", None if eigs_delta is None else f"{int(eigs_delta):+d}")

    order = load_heatmap_order()
    col_graph, col_heat = st.columns([3, 2])

    if selection["compare"] and selection["second_date"] is not None:
        second_date = selection["second_date"]
        second_idx = corr_index.get_loc(second_date)
        weights_2, thresh_2 = _weights_for(corr, second_idx, tau_value)
        pos_2 = _layout_for(weights_2, config, selection["fixed_layout"], layout)

        graph_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        draw_snapshot(ax1, thresh_t, pos_t, labels=config.data.symbols, title=date.strftime("%Y-%m-%d"))
        draw_snapshot(ax2, thresh_2, pos_2, labels=config.data.symbols, title=second_date.strftime("%Y-%m-%d"))
        col_graph.pyplot(graph_fig)

        heat_fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        draw_heatmap(bx1, corr[date_idx], order, labels=config.data.symbols, title=date.strftime("%Y-%m-%d"))
        draw_heatmap(bx2, corr[second_idx], order, labels=config.data.symbols, title=second_date.strftime("%Y-%m-%d"))
        col_heat.pyplot(heat_fig)
    else:
        graph_fig, ax = plt.subplots()
        draw_snapshot(ax, thresh_t, pos_t, labels=config.data.symbols, title=date.strftime("%Y-%m-%d"))
        col_graph.pyplot(graph_fig)

        heat_fig, hx = plt.subplots()
        draw_heatmap(hx, corr[date_idx], order, labels=config.data.symbols, title=date.strftime("%Y-%m-%d"))
        col_heat.pyplot(heat_fig)

    st.download_button(
        "Scarica snapshot del grafo (PDF)",
        data=_figure_pdf_bytes(graph_fig),
        file_name=f"grafo_{date.strftime('%Y-%m-%d')}.pdf",
        mime="application/pdf",
    )

    render_topology_strip(topology, events_list, date)

    st.divider()
    st.caption(
        f"τ calibrata = {tau.tau:.4f} · $T_w$ = {config.graph.window}g · "
        f"periodo {config.data.start.isoformat()} – {config.data.end.isoformat()} · "
        f"{len(config.data.symbols)} asset"
    )


if __name__ == "__main__":
    main()
