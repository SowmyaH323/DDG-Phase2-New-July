from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import (
    EnsembleWeights,
    load_models,
    parse_fasta,
    predict_one,
    prioritize_scan,
    scan_19aa_position,
    sequence_from_pdb,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
METADATA_PATH = APP_DIR / "phase2_metadata.json"

st.set_page_config(page_title="Phase-2 ΔΔG Predictor", page_icon="🧬", layout="wide")


@st.cache_resource(show_spinner="Loading trained models…")
def get_models():
    return load_models(ARTIFACTS_DIR, METADATA_PATH)


@st.cache_data(show_spinner=False)
def pdb_chains(pdb_bytes: bytes) -> list[str]:
    text = pdb_bytes.decode("utf-8", errors="ignore")
    chains: list[str] = []
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            chain = line[21].strip() or " "
            if chain not in chains:
                chains.append(chain)
    return chains or ["A"]


def save_uploaded_pdb(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".pdb"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()
    return Path(tmp.name)


def result_table(result: dict) -> pd.DataFrame:
    rows = []
    for key, label in (
        ("xgb", "XGBoost"),
        ("cnn", "CNN"),
        ("gnn", "GNN"),
        ("ensemble", "Ensemble"),
    ):
        value = result.get(key)
        if value is not None and pd.notna(value):
            rows.append(
                {
                    "Model": label,
                    "Predicted ΔΔG (kcal/mol)": round(float(value), 4),
                }
            )
    return pd.DataFrame(rows)


def show_single_result(result: dict) -> None:
    st.success(f"Prediction completed: {result['mutation']} — {result['ddg_class']}")
    st.dataframe(result_table(result), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Classification", result["ddg_class"])
    c2.metric("Confidence", result.get("confidence", "Not assigned"))

    if result.get("cnn_error"):
        st.warning(f"CNN unavailable for this input: {result['cnn_error']}")
    if result.get("gnn_error"):
        st.warning(f"GNN unavailable for this input: {result['gnn_error']}")

    output = pd.DataFrame([result])
    st.download_button(
        "Download prediction CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name=f"prediction_{result['mutation']}.csv",
        mime="text/csv",
    )


st.title("🧬 Phase-2 Mutation ΔΔG Predictor")
st.caption(
    "Frozen XGBoost + CNN + GNN inference pipeline. "
    "Negative values indicate predicted stabilization under the project convention."
)

try:
    models = get_models()
except Exception as exc:
    st.error(f"The trained models could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    temperature = st.number_input("Temperature (°C)", value=25.0, step=1.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    measurement_type = st.selectbox("Measurement type", models.metadata["measurement_types"])
    dataset = st.selectbox("Dataset encoding", models.metadata["datasets"])

    st.divider()
    st.subheader("Ensemble")
    custom_weights = st.toggle(
        "Advanced: custom exploratory weights",
        value=False,
        help=(
            "Default mode uses equal weights for XGBoost, CNN and GNN. "
            "Enable this only for exploratory testing."
        ),
    )

    if custom_weights:
        st.warning("Exploratory mode is active. The weights below are not the validated default.")
        wx_raw = st.slider("XGBoost", 0.0, 1.0, 0.34, 0.01)
        wc_raw = st.slider("CNN", 0.0, 1.0, 0.33, 0.01)
        wg_raw = st.slider("GNN", 0.0, 1.0, 0.33, 0.01)
        total = wx_raw + wc_raw + wg_raw
        if total <= 0:
            wx = wc = wg = 1.0 / 3.0
        else:
            wx, wc, wg = wx_raw / total, wc_raw / total, wg_raw / total
        ensemble_mode = "Custom exploratory"
    else:
        wx = wc = wg = 1.0 / 3.0
        ensemble_mode = "Validated equal-weight"

    weights = EnsembleWeights(wx, wc, wg)
    st.caption(
        f"{ensemble_mode}\n\n"
        f"XGBoost: {wx:.4f} · CNN: {wc:.4f} · GNN: {wg:.4f}"
    )

pdb_tab, fasta_tab = st.tabs(["PDB prediction", "FASTA prediction"])

with pdb_tab:
    st.subheader("Single-mutation prediction")
    uploaded = st.file_uploader("Upload a PDB file", type=["pdb"], key="pdb_file")

    if uploaded:
        chain_options = pdb_chains(uploaded.getvalue())
        chain_id = st.selectbox("Chain", chain_options, key="pdb_chain")
    else:
        chain_id = "A"

    mutation = st.text_input(
        "Mutation",
        placeholder="Example: N28I",
        help="Use one-letter amino-acid codes: wild type + residue number + mutant.",
        key="pdb_mutation",
    )

    if st.button(
        "Predict mutation",
        type="primary",
        disabled=uploaded is None or not mutation.strip(),
        key="pdb_predict",
    ):
        path = save_uploaded_pdb(uploaded)
        try:
            sequence = sequence_from_pdb(path, chain_id)
            result = predict_one(
                models,
                sequence,
                mutation,
                pdb_path=path,
                chain_id=chain_id,
                weights=weights,
                temperature=temperature,
                ph=ph,
                measurement_type=measurement_type,
                dataset=dataset,
            )
            show_single_result(result)
        except Exception as exc:
            st.error(str(exc))
        finally:
            path.unlink(missing_ok=True)

    st.divider()
    st.subheader("19-amino-acid saturation scan")
    st.caption("Upload the PDB once above, select a chain, and enter the residue position to test all 19 substitutions.")
    scan_position = st.number_input(
        "Residue position",
        min_value=1,
        value=28,
        step=1,
        key="pdb_scan_position",
    )

    if st.button(
        "Run 19-AA scan",
        disabled=uploaded is None,
        key="pdb_scan",
    ):
        path = save_uploaded_pdb(uploaded)
        try:
            sequence = sequence_from_pdb(path, chain_id)
            df = scan_19aa_position(
                models,
                sequence,
                int(scan_position),
                pdb_path=path,
                chain_id=chain_id,
                weights=weights,
                temperature=temperature,
                ph=ph,
                measurement_type=measurement_type,
                dataset=dataset,
            )
            df = prioritize_scan(df)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scan CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"PDB_position_{int(scan_position)}_19AA_scan.csv",
                mime="text/csv",
                key="pdb_scan_download",
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            path.unlink(missing_ok=True)

with fasta_tab:
    st.subheader("Single-mutation prediction")
    sequence_text = st.text_area(
        "Paste FASTA sequence or raw protein sequence",
        height=180,
        key="fasta_sequence",
    )
    mutation_fasta = st.text_input(
        "Mutation",
        placeholder="Example: N28I",
        key="fasta_mutation",
    )
    st.info("FASTA mode uses XGBoost only because CNN and GNN require a 3D structure.")

    if st.button(
        "Predict mutation",
        type="primary",
        disabled=not sequence_text.strip() or not mutation_fasta.strip(),
        key="fasta_predict",
    ):
        try:
            sequence = parse_fasta(sequence_text)
            result = predict_one(
                models,
                sequence,
                mutation_fasta,
                weights=weights,
                temperature=temperature,
                ph=ph,
                measurement_type=measurement_type,
                dataset=dataset,
            )
            show_single_result(result)
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("19-amino-acid saturation scan")
    fasta_scan_position = st.number_input(
        "Residue position",
        min_value=1,
        value=28,
        step=1,
        key="fasta_scan_position",
    )

    if st.button(
        "Run 19-AA scan",
        disabled=not sequence_text.strip(),
        key="fasta_scan",
    ):
        try:
            sequence = parse_fasta(sequence_text)
            df = scan_19aa_position(
                models,
                sequence,
                int(fasta_scan_position),
                weights=weights,
                temperature=temperature,
                ph=ph,
                measurement_type=measurement_type,
                dataset=dataset,
            )
            df = prioritize_scan(df)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scan CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"FASTA_position_{int(fasta_scan_position)}_19AA_scan.csv",
                mime="text/csv",
                key="fasta_scan_download",
            )
        except Exception as exc:
            st.error(str(exc))
