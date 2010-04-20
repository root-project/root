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
      *)  echo "Invalid option '$1'. Try $0 --help" ; exit 1 ;;
      esac
      shift
   done
fi

if [ "x$logfile" != "x" ] ; then
  cat $logfile
  if [ "x$testname" != "x" ] ; then
     echo "'root.exe -b -l -q $testname' exited with error code: $result" >> $logfile
  fi
fi
if [ "x$toremove" != "x" -a -e $toremove ] ; then 
  echo handleError.sh: '*** Deleting file' $toremove
  rm $toremove
fi
exit $result
