#!/usr/bin/env bash

# Refresh vendored upstream Magika sources and model assets.
set -euo pipefail

# Upstream sources and the exact subtrees we vendor.
MAGIKA_REPO_GIT="https://github.com/google/magika.git"
MAGIKA_REF="main"
MAGIKA_MODEL_NAME="standard_v3_3"
TEST_CORPUS_REPO_GIT="https://github.com/akunasoftware/test-corpus.git"
TEST_CORPUS_FIXTURES_PATH="file-classification/fixtures"
# Paths copied from the upstream repo into src/vendor/.
MAGIKA_VENDOR_PATHS=(
  "rust/lib/src/file.rs"
  "rust/lib/src/content.rs"
  "rust/lib/src/model.rs"
  "assets/models/${MAGIKA_MODEL_NAME}"
)

# Resolve local paths and make sure the temp clone is always removed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
MAGIKA_REPO_DIR="${TMP_DIR}/magika"
TEST_CORPUS_REPO_DIR="${TMP_DIR}/test-corpus"
VENDOR_DIR="${SCRIPT_DIR}/../src/vendor"
FIXTURES_DIR="${SCRIPT_DIR}/../fixtures"
trap 'rm -rf "${TMP_DIR}"' EXIT

# Shallow clones are enough because we only read tracked files.
git clone --depth 1 --branch "${MAGIKA_REF}" --single-branch \
  "${MAGIKA_REPO_GIT}" "${MAGIKA_REPO_DIR}" >/dev/null 2>&1

# Keep committed mod.rs, replace everything generated underneath it.
find "${VENDOR_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Recreate each vendored upstream path under src/vendor/.
for relative_path in "${MAGIKA_VENDOR_PATHS[@]}"
do
  mkdir -p "$(dirname "${VENDOR_DIR}/${relative_path}")"
  cp -R "${MAGIKA_REPO_DIR}/${relative_path}" "${VENDOR_DIR}/${relative_path}"
done

# Refresh local ignored fixtures from the shared corpus repo.
if ! git clone --depth 1 "${TEST_CORPUS_REPO_GIT}" "${TEST_CORPUS_REPO_DIR}" >/dev/null 2>&1 \
  || [ ! -d "${TEST_CORPUS_REPO_DIR}/${TEST_CORPUS_FIXTURES_PATH}" ]; then
  printf 'failed to locate shared fixtures at %s or %s\n' \
    "${TEST_CORPUS_REPO_GIT}" "${TEST_CORPUS_FIXTURES_PATH}" >&2
  exit 1
fi

rm -rf "${FIXTURES_DIR}"
mkdir -p "${FIXTURES_DIR}"
cp -R "${TEST_CORPUS_REPO_DIR}/${TEST_CORPUS_FIXTURES_PATH}/." "${FIXTURES_DIR}/"

# Keep the repo in its canonical formatted state after vendoring.
cargo fmt --all

# Emit a short summary so updates are traceable in logs.
printf 'Updated vendor assets from %s at %s using model %s\n' "${MAGIKA_REPO_GIT}" "${MAGIKA_REF}" "${MAGIKA_MODEL_NAME}"
printf 'Updated local fixtures from %s using %s\n' "${TEST_CORPUS_REPO_GIT}" "${TEST_CORPUS_FIXTURES_PATH}"
