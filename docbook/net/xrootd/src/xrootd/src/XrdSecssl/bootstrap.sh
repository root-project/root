#!/bin/sh
libtoolize --copy --force
aclocal
automake -acf
autoconf

