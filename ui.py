import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

import kibo_ra_v23 as kibo

auditor = kibo.KIBORA()


def _texts_from_inputs(text_input, file_input):
    if file_input is not None:
        return kibo.load_requirements(txt_path=file_input)
    if text_input and text_input.strip():
        return [line.strip() for line in text_input.splitlines() if line.strip()]
    return []


def run(text_input, file_input):
    texts = _texts_from_inputs(text_input, file_input)
    if not texts:
        raise gr.Error("Enter at least one requirement or upload a .txt file.")

    results = [auditor.assess(t) for t in texts]
    prefix = str(Path(tempfile.mkdtemp()) / "kibo_ra")

    xlsx_path, _, _ = kibo.save_outputs(results, prefix)

    df = kibo.results_dataframe(results)
    req_ids = df["requirement"].tolist()
    cobit_json = f"{prefix}_cobit_signals.json"
    cobit_csv = f"{prefix}_cobit_matrix.csv"
    kibo.save_governance_signals(results, req_ids, cobit_json, cobit_csv)

    heatmap_path = f"{prefix}_heatmap.png"
    kibo.save_heatmap(df, heatmap_path)

    cobit_df = pd.read_csv(cobit_csv)

    return df, xlsx_path, heatmap_path, heatmap_path, cobit_df, cobit_csv


with gr.Blocks(title="KIBO-RA") as demo:
    gr.Markdown("# KIBO-RA")
    gr.Markdown("Score requirements on performance, security, compliance, complexity, and ambiguity.")

    with gr.Row():
        text_input = gr.Textbox(lines=6, label="Requirements (one per line)")
        file_input = gr.File(label="...or upload a .txt file", file_types=[".txt"])

    run_btn = gr.Button("Run", variant="primary")

    gr.Markdown("## Results")
    results_table = gr.Dataframe(label="Scores")
    xlsx_download = gr.File(label="Download Excel")

    gr.Markdown("## Heatmap")
    heatmap_image = gr.Image(label="Heatmap")
    heatmap_download = gr.File(label="Download Heatmap")

    gr.Markdown("## COBIT Signals")
    cobit_table = gr.Dataframe(label="COBIT Objective Status")
    cobit_download = gr.File(label="Download COBIT Matrix (CSV)")

    run_btn.click(
        run,
        inputs=[text_input, file_input],
        outputs=[results_table, xlsx_download, heatmap_image, heatmap_download, cobit_table, cobit_download],
    )


if __name__ == "__main__":
    demo.launch()
