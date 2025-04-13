#!/bin/sh

# roottest needs several variables to be defined.
# Pushing them through the buildbot config is more difficult than just
# calculating them here. This script will thus invoke roottest for buildbot.
# Axel, 2010-03-25

# PWD on cygwin is garbled, need to adjust \cygwin/home to /home
if uname -a | grep -i cygwin > /dev/null; then
    PWD=${PWD##\\cygwin}
    cd $PWD
    export ROOTTEST_HOME="`cygpath -m $PWD`/"
else
    export ROOTTEST_HOME="$PWD/"
fi

STARTPWD=$PWD

# We might be building roottest for roottest-Ubuntu1004-64bit-nightly
# That wants to test ROOT-Ubuntu1004-64bit-nightly, so that's the ROOT
# version we need to set up.

# pwd is ..../ROOT-Ubuntu1004-64bit-nightly/build, so cd up:
cd ..
# and this is the slave that runs us:
BBARCH=`basename $PWD`
# this is the corresponding ROOT slave's location
BBARCH=../ROOT-${BBARCH#*-}/build
# we cd into its build directory and set ROOT up
cd  $BBARCH || (echo Cannot find directory $BBARCH from `pwd`; exit 1)
. bin/thisroot.sh || (echo Cannot find ROOT setup script in `pwd`; exit 1)
echo Set up ROOT in $ROOTSYS, SVN revision:
echo === svninfo.txt ===
cat $ROOTSYS/etc/svninfo.txt
echo === svninfo.txt ===

# cd back to where we started
cd $STARTPWD

# Make clean before making roottest, to not depend on dependencies:
#make clean "$@"
make -k FAST=1 "$@"
ret=$?
echo === svninfo.txt ===
cat $ROOTSYS/etc/svninfo.txt
echo === svninfo.txt ===
exit $ret
