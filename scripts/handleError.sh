#!/bin/sh

help() {
cat << EOF
'handleError.h' is a makefile helper that will print a log file generated during a Makefile
rule and still propagate the error.

Usage:     $0 [architecture] [flag=value]
EOF
}

if test $# -gt 0 ; then
   while test ! "x$1" = "x" ; do
      case "$1" in
      -*=*) optarg=`echo "$1" | sed 's/[-_a-zA-Z0-9]*=//'` ;;
      *) optarg= ;;
      esac

      case $1 in
      --help|-h) help ;   exit 0 ;;
      --result=*|--res=*) result=$optarg  ;;
      --log=*) logfile=$optarg  ;;
      --test=*) testname=$optarg  ;;
      --rm=*) toremove=$optarg  ;;
      --cmd=*) cmd=$optarg ;;
      *)  echo "Invalid option '$1'. Try $0 --help" ; exit 1 ;;
      esac
      shift
   done
fi

if [ "x$SUMMARY" != "x" ] ; then
   testname=`echo $testname | tr / .`
   SUMMARY_FILE=`echo $SUMMARY.$testname.summary | sed -e 's:/:_:g' `
fi

if [ "x$logfile" != "x" ] ; then
  if [ "x$SUMMARY" != "x" ] ; then 
     if [ "x$testname" != "x" ] ; then
        echo "--- FAILING TEST: make -C $CALLDIR $testname" > $SUMMARY_FILE
        cat $logfile >> $SUMMARY_FILE
        if [ `grep -c "exited with error code: $result" $logfile` -eq 0 ] ; then 
          if [ "x$cmd" = "xdiff" ] ; then 
            echo "diff command exited with error code: $result" >> $SUMMARY_FILE
          else 
            if [ "x$cmd" != "x" ] ; then 
              echo "'$cmd' exited with error code: $result" >> $SUMMARY_FILE
            else 
              echo "'A command like root.exe -b -l -q $testname' exited with error code: $result" >> $SUMMARY_FILE
            fi
          fi
        fi
     else 
        pid=$$
        # --- FAILING TEST: make -C dir/to/test $testname
        echo "--- FAILING TEST: make -C $CALLDIR test" > $SUMMARY.$pid.summary
        cat $logfile >> $SUMMARY.$pid.summary
     fi
  fi
  cat $logfile
  if [ "x$testname" != "x" ] ; then
     if [ "x$cmd" = "xdiff" ] ; then 
       echo "diff command exited with error code: $result" >> $logfile
     else 
       if [ "x$cmd" != "x" ] ; then 
         echo "'$cmd' exited with error code: $result" >> $logfile
       else 
         echo "'A command like root.exe -b -l -q $testname' exited with error code: $result" >> $logfile
       fi
     fi
  fi
fi
if [ "x$toremove" != "x" ] ; then
  if [ -e $toremove ] ; then 
    echo handleError.sh: '*** Deleting file' $toremove
    rm $toremove
  fi
fi
exit $result
