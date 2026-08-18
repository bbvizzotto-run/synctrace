#!/usr/bin/env bash
set -euo pipefail

# Downloads the public VidTIMIT source one archive at a time, as requested by
# the dataset maintainer. It does not download FakeAVCeleb or LipSyncTIMIT,
# which require separate approval from their respective maintainers.

ROOT_DIR="${1:-data/raw/vidtimit}"
ZIP_DIR="${ROOT_DIR}/zips"
EXTRACT_DIR="${ROOT_DIR}/extracted"
BASE_URL="https://zenodo.org/records/158963/files"

subjects=(
  fadg0 faks0 fcft0 fcmh0 fcmr0 fcrh0 fdac1 fdms0 fdrd1 fedw0 felc0 fgjd0
  fjas0 fjem0 fjre0 fjwb0 fkms0 fpkt0 fram1 mabw0 mbdg0 mbjk0 mccs0 mcem0
  mdab0 mdbb0 mdld0 mgwt0 mjar0 mjsw0 mmdb1 mmdm2 mpdf0 mpgl0 mrcz0 mreb0
  mrgg0 mrjo0 msjs1 mstk0 mtas1 mtmr0 mwbt0
)

mkdir -p "${ZIP_DIR}" "${EXTRACT_DIR}"

for subject in "${subjects[@]}"; do
  archive="${ZIP_DIR}/${subject}.zip"
  if [[ ! -s "${archive}" ]]; then
    echo "Downloading ${subject}.zip"
    curl --fail --location --retry 4 --retry-delay 3 --continue-at - \
      "${BASE_URL}/${subject}.zip?download=1" --output "${archive}"
  else
    echo "Using existing ${archive}"
  fi

  target="${EXTRACT_DIR}/${subject}"
  if [[ ! -d "${target}" ]]; then
    mkdir -p "${target}"
    unzip -q "${archive}" -d "${target}"
  fi
done

echo "VidTIMIT available at ${EXTRACT_DIR}"
