#!/bin/sh
#
# Author: Derek Feichtinger, 19 Oct 2005

# create autotools build files from the CVS sources
libtoolize --copy --force
aclocal
automake -acf
autoconf

