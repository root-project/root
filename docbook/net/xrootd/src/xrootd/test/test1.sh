#!/bin/bash
#################################################################
## simple test for xrootd build testing
##
## 
## Initial version: 31.5.2007
## by Derek Feichtinger <derek.feichtinger@cern.ch>
##
## Version info: $Id$
## Checked in by $Author$
################################################################################

#############################
# Bootstrap

test -r test/testconfig.sh && cd test

curdir=`pwd`
if test ! -r testconfig.sh ; then
    echo "Error: Cannot find configuration file testconfig.sh (pwd: $curdir)" >&2
    exit 1
fi

. testconfig.sh

echo cd $top_srcdir
top_srcdir=$(cd $top_srcdir;pwd)
top_builddir=$(cd $top_builddir;pwd)

if test ! -e $top_srcdir/src/XrdOfs/XrdOfs.hh; then
    echo "Error: $top_srcdir seems not to be the correct top_srcdir" >&2
    exit 1
fi

##################################################################
# CONFIGURATION SETTINGS

testname=test1
testdescr="test of a minimally configured xrootd"
xrdport=22000
testdir=$top_srcdir/test
workdir=$top_builddir/test/work
exportdir=$workdir/exportdir

logfile=$workdir/${testname}-xrootd.log
# name of the xrootd to avoid clashes with other running instances
# Warning: this strangely also affects the name of the logfile 
name=testxrootd$$
# logfile's true name is
reallogfile=$workdir/$name/test2-xrootd.log
##################################################################

cleanup() {
    if test 0"$1" -lt 2; then
	echo "Error: cleanup() no valid PID given: $1"
	exit 1
    fi
    kill $1 &>/dev/null
    status=$?
    if test 0"$status" -ne 0; then
	echo "Error: Failed to kill process ($PID)" >&2
	return 1
    fi
    return 0
}

cleanup_workarea() {
# clean up the work area
#    echo -n "Cleaning up work area..."
    files="$exportdir/testfile_r 
$exportdir/testfile_w
$workdir/testfile_r
$workdir/${testname}-xrootd.cfg
$workdir/$name/${testname}-xrootd.log"

    rm -f $files
    test -d $exportdir && rmdir $exportdir
    test -d $exportdir && echo "Warning: Failed to clean up $exportdir" >&2

    test -d $workdir/$name && rmdir $workdir/$name
    test -d $workdir/$name && echo "Warning: Failed to clean up $workdir/$name" >&2

    test -d $workdir && rmdir $workdir
    test -d $workdir && echo "Warning: Failed to clean up $workdir" >&2

    # why does this dir get produced?
    test -d $testdir/$name && rmdir $testdir/$name
}

##############################################################

echo "--------------------------------------------------------"
echo "TEST: $testname"
echo "Test description: $testdescr"
cd $testdir
cleanup_workarea
mkdir -p $workdir
mkdir -p $exportdir
cp $testdir/${testname}.sh $exportdir/testfile_r

# Create simple xrootd config file
echo "writing $workdir/test1-xrootd.cfg"
cat <<EOF > $workdir/${testname}-xrootd.cfg
xrootd.export $exportdir
xrd.port $xrdport
EOF

# start up xrootd
rm -rf $reallogfile
echo -n "Starting up a test xrootd (port $xrdport)... "
#echo $top_builddir/src/XrdXrootd/xrootd -c $workdir/${testname}-xrootd.cfg \
#      -n $name -l $logfile
$top_builddir/src/XrdXrootd/xrootd -c $workdir/${testname}-xrootd.cfg \
      -n $name -l $logfile &
PID=$!

if test 0"$PID" -eq 0; then
    echo "[FAILED]"
    echo "Error: Failed to get process ID of xrootd" >&2
    exit 1
fi

sleep 1

kill -0 $PID
status=$?
if test 0"$status" -ne 0; then
    echo "[FAILED]"
    echo "Error: Failed to start up test xrootd. Look at $reallogfile" >&2
    exit 1
fi
echo "[OK]"

# read a file
echo -n "Reading a file from xrootd..."
rm -f $workdir/testfile_r
$top_builddir/src/XrdClient/xrdcp -s root://localhost:$xrdport/$exportdir/testfile_r $workdir/testfile_r
if test ! -f $workdir/testfile_r; then
    echo "[FAILED]"
    cleanup $PID
    exit 1
fi
echo "[OK]"

# write a file
echo -n "Writing a file to xrootd..."
rm -f $exportdir/testfile_w
$top_builddir/src/XrdClient/xrdcp -s $workdir/testfile_r root://localhost:$xrdport/$exportdir/testfile_w
if test ! -f $exportdir/testfile_w; then
    echo "[FAILED]"
    cleanup $PID
    exit 1
fi
echo "[OK]"


# kill the xrootd process
echo -n "Shutting down the test xrootd..."
cleanup $PID
status=$?
#echo STATUS is $status
if test 0"$status" -ne 0; then
    echo "[FAILED]"
    echo "Error: Failed to kill process ($PID)" >&2
    exit 1
fi
echo "[OK]"

cleanup_workarea

exit 0




