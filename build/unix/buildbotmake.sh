#!/bin/bash

# Run configure if configure hasn't been run or if the configuration
# switches have changed.
# Axel, 2010-04-26

# Fix up windows vs unix path style:
cd $PWD

if ! test  -f config.status; then
    ./configure "$@" || exit $?
else
    ARGS="$@"
    while [ "x$1" != "x" ]; do
        if ! grep -e "$1" config.status > /dev/null 2>&1; then
            ./configure $ARGS || exit $?
            break
        fi
        shift
    done
fi

make -j2
