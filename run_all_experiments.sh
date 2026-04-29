#!/usr/bin/env bash
# Reproducible benchmark pipeline: Epidemic Source Detection (single-source)
# Experimental setup: Sterchi et al. (2025), Sections 5 & 6
#
# DATA PIPELINES
#   Static GNN  — continuous-time SIR C binary (main_static_sir.py)
#                 artifact: <dataset>_sir
#   Backtracking Network — discrete-time TSIR (main_tsir.py)
#                          artifact: <dataset>
#
# EXECUTION ORDER (strict — manages GPU/disk resources)
#   Phase 1 — Static GNN on standard datasets (malawi, france_office, lyon_ward, students)
#   Phase 2 — Backtracking Network on standard datasets
#   Phase 3 — Backtracking Network on escort (~14k nodes)
#   Phase 4 — Static GNN on escort (~14k nodes)
#
# OUTPUTS — clearly structured under figures/<dataset>/
#   figures/<dataset>/comparison/
#       rank_vs_outbreak_all.pdf        — rank vs. outbreak size, ALL methods together
#       top5_vs_outbreak_relative.pdf   — top-5 accuracy (fraction), ALL methods
#       top5_vs_outbreak_absolute.pdf   — top-5 hits (count), ALL methods
#   figures/<dataset>/static_gnn/       — per-method individual plots
#   figures/<dataset>/backtracking/     — per-method individual plots
#   figures/tables/benchmark_table.tex  — LaTeX benchmark table (all datasets)
#   figures/tables/benchmark_table.csv  — CSV version
#   logs/<dataset>_static_gnn.log       — full training + eval log
#   logs/<dataset>_backtracking.log     — full training + eval log
#   logs/<dataset>_manifest.txt         — run ID registry for viz scripts
#
# USAGE
#   ./run_all_experiments.sh                     # all phases
#   ./run_all_experiments.sh --phase 1           # one phase only
#   ./run_all_experiments.sh --skip-sir          # skip data generation
#   ./run_all_experiments.sh --phase 2 --skip-sir
#
# SIR CALIBRATION NOTE
#   β is set so R₀ = β×mean_deg/ν ≈ 2 (ν=1 fixed).
#   T targets ~40% infected.  If avg_outbreak logged in Stage 1 is outside
#   0.35–0.45, adjust T in the dataset's static_sir.yml or tsir.yml and
#   re-run Stage 1 without --skip-sir.

set -euo pipefail

SKIP_SIR=false
PHASE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sir|--skip-tsir) SKIP_SIR=true ;;
    --phase) PHASE="$2"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

mkdir -p logs

STANDARD=(malawi france_office lyon_ward students)

# ── Helper: extract W&B run ID from captured output ───────────────────────────
_extract_run_id() { echo "$1" | grep -oP "/runs/\K[a-z0-9]+" | head -1 || true; }

# ── Helper: append one manifest entry ─────────────────────────────────────────
_manifest_append() {
  local MANIFEST="$1" KIND="$2" MODEL="$3" RUN_ID="$4" PIPELINE="$5"
  if [[ -z "${RUN_ID}" ]]; then return; fi
  if [[ "${KIND}" == "eval_run" ]]; then
    printf "eval_run run_id=%s pipeline=%s\n" "${RUN_ID}" "${PIPELINE}" >> "${MANIFEST}"
  else
    printf "model=%s run_id=%s pipeline=%s\n" "${MODEL}" "${RUN_ID}" "${PIPELINE}" >> "${MANIFEST}"
  fi
}

# ── run_static_gnn <dataset> ──────────────────────────────────────────────────
# Phase 1 / Phase 4 helper.
# Runs: static SIR → Static GNN training → baselines on static SIR data.
# Captures run IDs and writes to logs/<dataset>_manifest.txt.
# Generates per-dataset comparison figures when done.
run_static_gnn() {
  local NWK="$1"
  local SIR_ART="${NWK}_sir"
  local MANIFEST="logs/${NWK}_manifest.txt"
  local LOG="logs/${NWK}_static_gnn.log"
  local FIG_DIR="figures/${NWK}"

  echo ""
  echo "============================================================"
  echo " Static GNN | ${NWK}  ($(date '+%Y-%m-%d %H:%M'))"
  echo " Artifact  : ${SIR_ART}"
  echo " Log       : ${LOG}"
  echo " Figures   : ${FIG_DIR}/comparison/"
  echo "============================================================"

  # Initialise manifest (create or clear the static section)
  grep -v "pipeline=static" "${MANIFEST}" 2>/dev/null > "${MANIFEST}.tmp" || true
  mv "${MANIFEST}.tmp" "${MANIFEST}" 2>/dev/null || : > "${MANIFEST}"

  # ── Stage 1: Static SIR simulation ──────────────────────────────────────
  if [ "${SKIP_SIR}" = false ]; then
    echo "" | tee -a "${LOG}"
    echo "=== Stage 1: Static SIR simulation ===" | tee -a "${LOG}"
    python main_static_sir.py \
      --cfg "exp/${NWK}/static_sir.yml" \
      --data "${SIR_ART}" \
      2>&1 | tee -a "${LOG}"
  else
    echo "(--skip-sir: reusing artifact '${SIR_ART}:latest')" | tee -a "${LOG}"
  fi

  # ── Stage 2: Static GNN training ────────────────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 2: Static GNN training ===" | tee -a "${LOG}"
  GNN_OUT=$(python main_train.py \
    --cfg "exp/${NWK}/static_gnn.yml" \
    --data "${SIR_ART}:latest" \
    2>&1 | tee /dev/stderr)
  printf "%s\n" "${GNN_OUT}" >> "${LOG}"
  GNN_RUN=$(_extract_run_id "${GNN_OUT}")
  echo "  Static GNN run ID : ${GNN_RUN:-unknown}" | tee -a "${LOG}"
  _manifest_append "${MANIFEST}" "model" "static_gnn" "${GNN_RUN}" "static"

  # ── Stage 3: Baselines on static SIR data ───────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 3: Baselines (static SIR data, n_truth=100) ===" | tee -a "${LOG}"
  EVAL_OUT=$(python main_eval.py \
    --cfg "exp/${NWK}/eval.yml" \
    --data "${SIR_ART}:latest" \
    --override eval.n_truth=100 eval.min_outbreak=1 \
    2>&1 | tee /dev/stderr)
  printf "%s\n" "${EVAL_OUT}" >> "${LOG}"
  EVAL_RUN=$(_extract_run_id "${EVAL_OUT}")
  echo "  Eval run ID       : ${EVAL_RUN:-unknown}" | tee -a "${LOG}"
  _manifest_append "${MANIFEST}" "eval_run" "" "${EVAL_RUN}" "static"

  # ── Stage 4: Figures ────────────────────────────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 4: Comparison figures ===" | tee -a "${LOG}"
  python viz/compare_all.py \
    --manifest  "${MANIFEST}" \
    --dataset   "${NWK}" \
    --output-dir "${FIG_DIR}/comparison" \
    --pipeline  static \
    --per-method \
    2>&1 | tee -a "${LOG}"

  echo "" | tee -a "${LOG}"
  echo "  Done: ${NWK} (Static GNN)" | tee -a "${LOG}"
}

# ── run_backtracking <dataset> ────────────────────────────────────────────────
# Phase 2 / Phase 3 helper.
# Runs: TSIR → Backtracking Network training → baselines on TSIR data.
run_backtracking() {
  local NWK="$1"
  local TSIR_ART="${NWK}"
  local MANIFEST="logs/${NWK}_manifest.txt"
  local LOG="logs/${NWK}_backtracking.log"
  local FIG_DIR="figures/${NWK}"

  echo ""
  echo "============================================================"
  echo " Backtracking Network | ${NWK}  ($(date '+%Y-%m-%d %H:%M'))"
  echo " Artifact  : ${TSIR_ART}"
  echo " Log       : ${LOG}"
  echo " Figures   : ${FIG_DIR}/comparison/"
  echo "============================================================"

  # Extend (not clear) the manifest so static entries survive if Phase 1 ran first
  grep -v "pipeline=temporal" "${MANIFEST}" 2>/dev/null > "${MANIFEST}.tmp" || true
  mv "${MANIFEST}.tmp" "${MANIFEST}" 2>/dev/null || : > "${MANIFEST}"

  # ── Stage 1: TSIR simulation ─────────────────────────────────────────────
  if [ "${SKIP_SIR}" = false ]; then
    echo "" | tee -a "${LOG}"
    echo "=== Stage 1: TSIR simulation ===" | tee -a "${LOG}"
    python main_tsir.py \
      --cfg "exp/${NWK}/tsir.yml" \
      --data "${TSIR_ART}" \
      2>&1 | tee -a "${LOG}"
  else
    echo "(--skip-sir: reusing artifact '${TSIR_ART}:latest')" | tee -a "${LOG}"
  fi

  # ── Stage 2: Backtracking Network training ───────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 2: Backtracking Network training ===" | tee -a "${LOG}"
  BN_OUT=$(python main_train.py \
    --cfg "exp/${NWK}/backtracking.yml" \
    --data "${TSIR_ART}:latest" \
    2>&1 | tee /dev/stderr)
  printf "%s\n" "${BN_OUT}" >> "${LOG}"
  BN_RUN=$(_extract_run_id "${BN_OUT}")
  echo "  BN run ID         : ${BN_RUN:-unknown}" | tee -a "${LOG}"
  _manifest_append "${MANIFEST}" "model" "backtracking" "${BN_RUN}" "temporal"

  # ── Stage 3: Baselines on TSIR data ─────────────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 3: Baselines (TSIR data) ===" | tee -a "${LOG}"
  EVAL_OUT=$(python main_eval.py \
    --cfg "exp/${NWK}/eval.yml" \
    --data "${TSIR_ART}:latest" \
    --override eval.min_outbreak=1 \
    2>&1 | tee /dev/stderr)
  printf "%s\n" "${EVAL_OUT}" >> "${LOG}"
  EVAL_RUN=$(_extract_run_id "${EVAL_OUT}")
  echo "  Eval run ID       : ${EVAL_RUN:-unknown}" | tee -a "${LOG}"
  _manifest_append "${MANIFEST}" "eval_run" "" "${EVAL_RUN}" "temporal"

  # ── Stage 4: Figures ────────────────────────────────────────────────────
  echo "" | tee -a "${LOG}"
  echo "=== Stage 4: Comparison figures ===" | tee -a "${LOG}"
  python viz/compare_all.py \
    --manifest   "${MANIFEST}" \
    --dataset    "${NWK}" \
    --output-dir "${FIG_DIR}/comparison" \
    --pipeline   temporal \
    --per-method \
    2>&1 | tee -a "${LOG}"

  echo "" | tee -a "${LOG}"
  echo "  Done: ${NWK} (Backtracking Network)" | tee -a "${LOG}"
}

# ── Phase execution ───────────────────────────────────────────────────────────

if [[ "${PHASE}" == "all" || "${PHASE}" == "1" ]]; then
  echo ""
  echo "################################################################"
  echo "# PHASE 1 — Static GNN on standard datasets"
  echo "# Networks  : ${STANDARD[*]}"
  echo "# SIR model : continuous-time C binary, R₀≈2, ~40% infected"
  echo "# Training  : 500 MC/node, 70/30 split, patience=5, 3 reps"
  echo "################################################################"
  for NWK in "${STANDARD[@]}"; do
    run_static_gnn "${NWK}"
  done
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "2" ]]; then
  echo ""
  echo "################################################################"
  echo "# PHASE 2 — Backtracking Network on standard datasets"
  echo "# Networks  : ${STANDARD[*]}"
  echo "# SIR model : discrete-time TSIR, full temporal contact sequence"
  echo "# Training  : 500 MC/node, 80/20 split, patience=30, 3 reps"
  echo "################################################################"
  for NWK in "${STANDARD[@]}"; do
    run_backtracking "${NWK}"
  done
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "3" ]]; then
  echo ""
  echo "################################################################"
  echo "# PHASE 3 — Backtracking Network on escort (~14k nodes)"
  echo "# WARNING: high compute + disk; see exp/escort/tsir.yml notes"
  echo "################################################################"
  run_backtracking "escort"
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "4" ]]; then
  echo ""
  echo "################################################################"
  echo "# PHASE 4 — Static GNN on escort (~14k nodes)"
  echo "# WARNING: ~13 GB disk; n_runs=10 (limited); see static_sir.yml"
  echo "################################################################"
  run_static_gnn "escort"
fi

# ── Final: benchmark table across all datasets ────────────────────────────────

if [[ "${PHASE}" == "all" ]]; then
  echo ""
  echo "################################################################"
  echo "# Final: Benchmark table (all datasets)"
  echo "################################################################"
  ALL_DATASETS="${STANDARD[*]} escort"
  python -m eval.tables benchmark \
    --data ${ALL_DATASETS} \
    --output figures/tables/ \
    --metrics top_5 error_dist mrr cred_set_size_90 resistance \
    2>&1 | tee logs/benchmark_table.log || echo "  (W&B table skipped — check logs/benchmark_table.log)"
fi

echo ""
echo "################################################################"
echo " Pipeline complete."
echo ""
echo " Output directory structure:"
echo "   figures/"
echo "     <dataset>/"
echo "       comparison/"
echo "         rank_vs_outbreak_all.pdf      ← ALL methods on one plot"
echo "         top5_vs_outbreak_relative.pdf ← Top-5 % by outbreak size"
echo "         top5_vs_outbreak_absolute.pdf ← Top-5 count by outbreak size"
echo "       <method>/"
echo "         rank_vs_outbreak.pdf          ← per-method with scatter"
echo "         top5_vs_outbreak.pdf"
echo "     tables/"
echo "         benchmark_table.tex           ← LaTeX table"
echo "         benchmark_table.csv"
echo ""
echo "   logs/"
echo "     <dataset>_manifest.txt            ← run ID registry"
echo "     <dataset>_static_gnn.log          ← full training log"
echo "     <dataset>_backtracking.log"
echo ""
echo " To regenerate figures from existing runs (no retraining):"
echo "   python viz/compare_all.py \\"
echo "     --manifest logs/<dataset>_manifest.txt \\"
echo "     --dataset <dataset> \\"
echo "     --output-dir figures/<dataset>/comparison \\"
echo "     --per-method"
echo "################################################################"
