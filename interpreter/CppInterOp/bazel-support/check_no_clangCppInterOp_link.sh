#!/bin/bash
# Fail if $1 links libclangCppInterOp: the dispatch tests must reach it only
# via dlopen(RTLD_LOCAL), proving symbol isolation.
ldd "$1" | grep -q libclangCppInterOp && exit 1 || exit 0
