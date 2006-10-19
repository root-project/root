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

host=`hostname -s`
dir=`dirname $0`

# The config is expected to set CINTSYSDIR
# and CONFIG_PLATFORM if necessary
. $dir/run_cinttest.$host.config

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.
export PATH=${CINTSYSDIR}:${CINTSYSDIR}/test:${PATH}
export CVS_RSH=ssh

error_handling() {
    cd $CINTSYSDIR
    write_summary
    upload_log summary.log cint_

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
    exit $1
}

upload_log() {    
    target_name=$2$1.$host
    scp $1 flxi02:/afs/.fnal.gov/files/expwww/root/html/roottest/$target_name > scp.log 2>&1 
}

write_summary() {
   lline="<td style=\"width: 100px; text-align: center;\" >"
   rline="</td>"
         osline="$lline $OSNAME $rline"
        cvsline="$lline <a href="cint_cvsupdate.log.$host">$cvsstatus</a>           $rline"
  configureline="$lline <a href="cint_configure.log.$host">$configurestatus</a>     $rline"
      gmakeline="$lline <a href="cint_gmake.log.$host">$mainstatus</a></td>         $rline"
       testline="$lline <a href="cint_testall.log.$host">$teststatus</a>            $rline"
   testdiffline="$lline <a href="cint_testdiff.log.$host">$teststatus</a>           $rline"
#     stressline="$lline <a href="speedresult.log.$host">$teststatus</a>            $rline"
   date=`date +"%b %d %Y"`
   dateline="$lline $date $rline"
   
   echo $osline       >  $CINTSYSDIR/summary.log
   echo $cvsline      >> $CINTSYSDIR/summary.log
   echo $configureline>> $CINTSYSDIR/summary.log
   echo $gmakeline    >> $CINTSYSDIR/summary.log
   echo $testline     >> $CINTSYSDIR/summary.log
   echo $testdiffline >> $CINTSYSDIR/summary.log
#  echo $stressline   >> $CINTSYSDIR/summary.log
   echo $dateline     >> $CINTSYSDIR/summary.log
}

na="N/A"
success="Ok."
failure="Failed"
cvsstatus=$na
configurestatus=$na
mainstatus=$na
teststatus=$na

cd $CINTSYSDIR
echo "Will build CINT in $host:$PWD"

cvs -z9 -q update -dAP > cvsupdate.log
result=$?
if test $result != 0; then 
    cvsstatus=$failure
else
    cvsstatus=$success
fi
upload_log cvsupdate.log cint_

if [ "x$Setup" = "xyes" ] ; then
  ./configure $CONFIG_PLATFORM > configure.log 2>&1
  result=$?
  upload_log configure.log cint_
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

upload_log gmake.log cint_
if test $result != 0; then 
   mainstatus=$failure
   error_handling $result "CINT's gmake failed!  See log file at $CINTSYSDIR/gmake.log"
fi
mainstatus=$success
echo "CINT's gmake succeeded"

echo "Will build the cintdlls"
$MAKE -k dlls >> gmake.log 2>&1
result=$?
upload_log gmake.log cint_
if test $result != 0; then 
   mainstatus=$failure
   error_handling $result "CINT's gmake dlls failed!  See log file at $CINTSYSDIR/gmake.log"
fi
mainstatus=$success
echo "CINT's gmake dlls succeeded"

#if test $result != 0; then 
#   echo "CINT's gmake failed!  See log file at $CINTSYSDIR/gmake.log"
#   exit $result
#fi
#echo "CINT's gmake succeeded"

#cd test
echo "Will run CINT test in $PWD"
time $MAKE test < /dev/null > testall.log 2>&1
result=$?
echo The expected time were real=$REALTIME user=$USERTIME | tee -a testall.log

upload_log testall.log cint_
if test $result != 0; then
   teststatus=$failure
   error_handling $result "CINT's test failed!  See log file at $CINTSYSDIR/testall.log"
fi

cd test
echo 'diff testdiff.ref testdiff.txt' > testdiff.log
diff testdiff.ref testdiff.txt >> testdiff.log
result=$?

upload_log testdiff.log cint_
if test $result != 0; then
   teststatus=$failure
   error_handling $result "CINT's test failed!  See log file at $CINTSYSDIR/testall.log"
fi
teststatus=$success
echo "CINT test succeeded"

cd $CINTSYSDIR
write_summary
upload_log summary.log cint_
