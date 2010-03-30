#!/bin/bash

[ -e config/Makefile.config ] || ./configure "$@"
if [ $? -ne 0 ]; then
    echo 'ERROR running configure, bailing out before building!' >&2
    exit $?
fi
make -j2
