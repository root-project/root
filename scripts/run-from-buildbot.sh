#!/bin/bash

# roottest needs several variables to be defined.
# Pushing them through the buildbot config is more difficult than just
# calculating them here. This script will thus invoke roottest for buildbot.

OLDPWD=$PWD

# We might be building roottest for roottest-Ubuntu1004-64bit-nightly
# That wants to test ROOT-Ubuntu1004-64bit-nightly, so that's the ROOT
# version we need to set up.

# pwd is ..../ROOT-Ubuntu1004-64bit-nightly/build, so cd down:
cd ..
# and this is the slave that runs us:
BBARCH=`basename $PWD`
# this is the corresponding ROOT slave's location
BBARCH=../ROOT-${BBARCH#*-}/build
# we cd into its build directory and set ROOT up
cd  $BBARCH || (echo Cannot find directory $BBARCH from `pwd`; exit 1)
. bin/thisroot.sh || (echo Cannot find ROOT setup script in `pwd`; exit 1)

# cd back to where we started
cd $OLDPWD

# some more env vars
# Did Philippe fix it? export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
export ROOTTEST_HOME=$PWD

# Make clean before making roottest, to not depend on dependencies:
make clean -j4
# Forward arguments to make:
make "$@"
