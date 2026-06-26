#!/usr/bin/env bash

set -e

echo "Running $(realpath $0) from $PWD"

WORKSPACE="${GITHUB_WORKSPACE:-${PWD}}"
echo "Setting WORKSPACE to $WORKSPACE"

SRC_DIR="${WORKSPACE}/ROOT-CI/src"
BUILD_DIR="${WORKSPACE}/ROOT-CI/build"

mkdir -v -p "${SRC_DIR}" "${BUILD_DIR}"

cmake -B "${BUILD_DIR}" -S ${SRC_DIR} \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
  -Dminimal=on -Dasserts=on

clang-tidy --version
exit 0
