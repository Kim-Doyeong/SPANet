#!/bin/bash
# usage: ./genOnnx.sh <VERSION>

set -e

# -------- colors --------
BLUE="\033[1;34m"
GREEN="\033[1;32m"
RED="\033[1;31m"
NC="\033[0m"   # no color
# ------------------------

VERSION=$1
OUTDIR="spanet_output/version_${VERSION}"
OPTIONS_JSON="${OUTDIR}/options.json"

if [[ -z "$VERSION" ]]; then
  echo -e "${RED}ERROR:${NC} Usage: $0 <version>"
  exit 1
fi

if [[ ! -f "$OPTIONS_JSON" ]]; then
  echo -e "${RED}ERROR:${NC} ${OPTIONS_JSON} not found"
  exit 1
fi

# training_file 추출 (jq 없이)
TRAINING_FILE=$(
  grep '"training_file"' "$OPTIONS_JSON" \
  | sed -E 's/.*"training_file"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'
)

if [[ -z "$TRAINING_FILE" ]]; then
  echo -e "${RED}ERROR:${NC} training_file not found in options.json"
  exit 1
fi

echo -e "${BLUE}[INFO]${NC} Using training file:"
echo -e "${BLUE}[INFO]${NC} ${TRAINING_FILE}"

# ------------------------
# ONNX export
# ------------------------
echo -e "${GREEN}▶ python -m spanet.export ./${OUTDIR} ${OUTDIR}/spanet.onnx${NC}"
python -m spanet.export "./${OUTDIR}" "${OUTDIR}/spanet.onnx"

echo -e "${GREEN}▶ python -m spanet.export ./${OUTDIR} ${OUTDIR}/spanetlast.onnx --checkpoint ...${NC}"
python -m spanet.export "./${OUTDIR}" "${OUTDIR}/spanetlast.onnx" \
  --checkpoint "${OUTDIR}/checkpoints/last.ckpt"

echo -e "${BLUE}▶./run_spanet_and_overlay.sh ${VERSION} 500000 semileptonic${NC}"

# ------------------------
# test (best)
# ------------------------
echo -e "${GREEN}▶ python -m spanet.test ./${OUTDIR} -tf ${TRAINING_FILE}${NC}"
python -m spanet.test "./${OUTDIR}" -tf "${TRAINING_FILE}" \
  > "${OUTDIR}/testlog.txt"


echo -e "${GREEN}▶ head -n 26 ${OUTDIR}/testlog.txt${NC}"
head -n 26 "${OUTDIR}/testlog.txt"

echo -e "${GREEN}▶ tail ${OUTDIR}/testlog.txt${NC}"
tail "${OUTDIR}/testlog.txt"


# ------------------------
# test (last)
# ------------------------
echo -e "${BLUE}[INFO]${NC} LAST CHECKPOINT"

echo -e "${GREEN}▶ python -m spanet.test ./${OUTDIR} -tf ${TRAINING_FILE} --checkpoint last.ckpt${NC}"
python -m spanet.test "./${OUTDIR}" -tf "${TRAINING_FILE}" \
  --checkpoint "${OUTDIR}/checkpoints/last.ckpt" \
  > "${OUTDIR}/testlog_last.txt"

echo -e "${GREEN}▶ head -n 26 ${OUTDIR}/testlog_last.txt${NC}"
head -n 26 "${OUTDIR}/testlog_last.txt"

echo -e "${GREEN}▶ tail ${OUTDIR}/testlog_last.txt${NC}"
tail "${OUTDIR}/testlog_last.txt"

#------------------------
# run comparision
#------------------------
echo -e "${GREEN}▶ python scripts/plot_purity_vs_version.py${NC}"
python scripts/plot_purity_vs_version.py
