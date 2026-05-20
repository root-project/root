#!/bin/sh

Setup=yes
while test "x$1" != "x"; do
   case $1 in 
      "-v" ) verbose=x; shift ;;
      "-q" ) Setup=no; shift ;;
      "-c" ) clean=yes; shift ;;
      "-h" ) help=x; shift;;
      *) break;;
   esac
done

if test "x$help" != "x"; then
    echo "$0 [options]"
    echo "Option:"
    echo "  -v : verbose"
    echo "  -q : skip the setup step"
    exit
fi

if [ "x$verbose" = "xx" ] ; then
   set -x
fi 

MAKE=gmake
REALTIME=n/a
USERTIME=n/a
UPLOAD_LOCATION=flxi06.fnal.gov:/afs/.fnal.gov/files/expwww/root/html/roottest/
UPLOAD_SYNC="ssh -x flxi06.fnal.gov bin/flush_webarea"

host=`hostname -s`
dir=`dirname $0`

# The config is expected to set CINTSYSDIR
# and CONFIG_PLATFORM if necessary
. $dir/run_cinttest.$host.config

ulimit -t 3600

upload_sync() {
    if [ "x$UPLOAD_SYNC" != "x" ] ; then
       eval $UPLOAD_SYNC
    fi
}

error_handling() {
    cd $CINTSYSDIR
    write_summary
    upload_log summary.log ${CORE}cint_

    echo "Found an error on \"$host\" ("`uname`") in $ROOTLOC"
    echo "Error: $2"
    echo "See full log file at http://www-root.fnal.gov/roottest/cint_summary.shtml"

    if [ "x$mail" = "xx" ] ; then
	mail -s "root $OSNAME test" $mailto <<EOF
Failure while building CINT on $host ("`uname`") in CINTSYSDIR
Error: $2
See full log file at http://www-root.fnal.gov/roottest/cint_summary.shtml
EOF
    fi
    upload_sync
    exit $1
}

upload_log() {    
    target_name=$2$1.$host
    scp $1 $UPLOAD_LOCATION/cint-today/$target_name > scp.log 2>&1 
    if test $result != 0; then 
        cat scp.log 
    fi
}

na="N/A"
success="Ok."
failure="Failed"
cvsstatus=$na
configurestatus=$na
mainstatus=$na
teststatus=$na
gmaketeststatus=$na

one_line() {
   ref=$1
   status=$2

   sline="<td style=\"width: 100px; background:lime; text-align: center;\" >"
   nline="<td style=\"width: 100px; background:gray; text-align: center;\" >"
   fline="<td style=\"width: 100px; background:orange; text-align: center;\" >"
   rline="</td>"

   if test "x$status" = "x$success"; then
      line="$sline <a href="cint-today/$ref">$status</a>           $rline"
   elif test "x$status" = "x$na"; then
      line="$nline <a href="cint-today/$ref">$status</a>           $rline"
   else
      line="$fline <a href="cint-today/$ref">$status</a>           $rline"
   fi
   echo $line
}

write_summary() {
   lline="<td style=\"width: 100px; text-align: center;\" >"
   rline="</td>"
         osline="$lline$OSNAME ${CORE}$rline"
        cvsline=`one_line ${CORE}cint_cvsupdate.log.$host $cvsstatus`
  configureline=`one_line ${CORE}cint_configure.log.$host $configurestatus`
      gmakeline=`one_line ${CORE}cint_gmake.log.$host $mainstatus`
       testline=`one_line ${CORE}cint_testall.log.$host $teststatus`
   testdiffline=`one_line ${CORE}cint_testdiff.log.$host $teststatus`

   date=`date +"%b %d %Y"`
   dateline="$lline $date $rline"
   
   echo $osline       >  $CINTSYSDIR/summary.log
   echo $cvsline      >> $CINTSYSDIR/summary.log
   echo $configureline>> $CINTSYSDIR/summary.log
   echo $gmakeline    >> $CINTSYSDIR/summary.log
   echo $testline     >> $CINTSYSDIR/summary.log
   echo $testdiffline >> $CINTSYSDIR/summary.log
   echo $dateline     >> $CINTSYSDIR/summary.log
}

runtest() {
  opt=$1
  
  #cd test
  echo "Will run CINT test in $PWD with $opt"
  time TESTFLAGS="$TESTFLAGS $opt" $MAKE test < /dev/null > test.log 2>&1
  result=$?
  echo The expected time were real=$REALTIME user=$USERTIME | tee -a test.log

  echo "Will run CINT test in $PWD with $opt" >> testall.log
  cat test.log >> testall.log
  upload_log testall.log ${CORE}cint_
  gmake_result=$result
  if test $result != 0; then
     teststatus=$failure
     #error_handling $result "CINT's test failed the gmake!  See log file at $CINTSYSDIR/testall.log"
  fi

  eval export `grep G__CFG_ARCH Makefile.conf | sed -e 's/ := /=/'`
  cd test
  echo "Run diff of test output in $PWD with $opt" >> testdiff.log
  echo "diff -ub testdiff.${CORE}${G__CFG_ARCH}.ref testdiff.txt" >> testdiff.log
  diff -ub testdiff.${CORE}${G__CFG_ARCH}.ref testdiff.txt >> testdiff.log
  result=$?

  upload_log testdiff.log ${CORE}cint_

  if test $gmake_result != 0; then
     error_handling $result "CINT's test failed the gmake!  See log file at $CINTSYSDIR/testall.log"
  fi
  if test $result != 0; then
     teststatus=$failure
     error_handling $result "CINT's test failed the diff!  See log file at $CINTSYSDIR/testall.log"
  fi
  teststatus=$success
  echo "CINT test succeeded"

  cd ..
}

runbuild() {

  export TOPDIR=$1
  export CORE=$2
  export CINTSYSDIR=${TOPDIR}/${CORE}cint

  export LD_LIBRARY_PATH=${CINTSYSDIR}/lib:${LD_LIBRARY_PATH}:.
  export PATH=${CINTSYSDIR}/bin:${CINTSYSDIR}/test:${PATH}

  if [ "x$CORE" = "xnew" ] ; then
     CONFIG_CORE=--coreversion=new
  else
     CONFIG_CORE=--coreversion=old
  fi
  export TESTFLAGS="--hide -k"
     
  cd ${TOPDIR}  
  echo "Will build ${CORE} CINT in $host:${CINTSYSDIR}"

  svn co http://root.cern.ch/svn/root/trunk/cint $CINTSYSDIR > cvsupdate.log
  result=$?
  if test $result != 0; then 
    cvsstatus=$failure
  else
    cvsstatus=$success
  fi
  upload_log cvsupdate.log ${CORE}cint_

  cd $CINTSYSDIR

  if [ "x$Setup" = "xyes" ] ; then
     ./configure $CONFIG_CORE $CONFIG_PLATFORM  > configure.log 2>&1
     result=$?
     upload_log configure.log ${CORE}cint_
     if test $result != 0; then
        configurestatus=$failure
        error_handling $result "CINT's configure failed!  See log file at $CINTSYSDIR/configure.log"
     fi
     echo "CINT's configure succeeded"
  fi
  configurestatus=$success

  if [ "x$clean" = "xyes" ] ; then
     gmake clean
  fi

  $MAKE > gmake.log 2>&1
  result=$?

  upload_log gmake.log ${CORE}cint_
  if test $result != 0; then 
     mainstatus=$failure
     error_handling $result "CINT's gmake failed!  See log file at $CINTSYSDIR/gmake.log"
  fi
  mainstatus=$success
  echo "CINT's gmake succeeded"

  upload_log gmake.log ${CORE}cint_
  mainstatus=$success
  echo "CINT's gmake dlls succeeded"

  rm -f test/testdiff.log testall.log
  runtest "-c -O0"
  runtest "-c -O1"
  runtest "-c -O2"
  runtest "-c -O3"
  runtest "-c -O4"
 
  cd $CINTSYSDIR
  write_summary
  upload_log summary.log ${CORE}cint_
}

TOPDIR=`dirname $CINTSYSDIR`
runbuild $TOPDIR ""
# runbuild $TOPDIR new

upload_sync
