#!/usr/bin/env bash

set -e

echo "Running $(realpath $0) from $PWD"

WORKSPACE="${GITHUB_WORKSPACE:-${PWD}}"
echo "Setting WORKSPACE to $WORKSPACE"

SRC_DIR="${WORKSPACE}/ROOT-CI/src"
BUILD_DIR="${WORKSPACE}/ROOT-CI/build"

mkdir -v -p "$SRC_DIR" "$BUILD_DIR"

cmake -B "$BUILD_DIR" -S $SRC_DIR \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
  -Dminimal=on -Dasserts=on

clang-tidy --version

HAVE_RUN=0
EXIT_CODE=0
for f in $(find "$SRC_DIR" -mindepth 2 -not -path '*/interpreter/*' -not -path '*/roofit/*' -name .clang-tidy); do
   set -x
   run-clang-tidy -use-color 1 -j $(nproc) -p "$BUILD_DIR" -config-file "$f" "$(dirname $f)/" || EXIT_CODE=1
   set +x
   HAVE_RUN=1
done

if [ $HAVE_RUN -eq 0 ]; then
   echo "clang-tidy ran nowhere; that doesn't seem right."
   exit 1
fi

exit $EXIT_CODE
