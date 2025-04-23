#!/bin/sh

vers=1
dict=c

while test "x$1" != "x"; do
   case $1 in 
      "-s" ) shift; vers=$1; shift ;;
      "-d" ) shift; dict=$1; shift ;;
      "-h" ) help=x; shift;;
      *) help=x; shift;;
   esac
done

if [ "x$help" = "xx" ]; then
   echo "compare.sh [-s schema_version] [-d dictionary_type]"
   exit 0
fi

if [ "x$dict" = "xc" ]; then 
   dict=
fi

files=`ls ${dict}test*_rv${vers}*.log`

for f in $files ; 
do
   wf=`echo $f | sed -e 's/_[^_]*\.log/_wv1.log/' `
   echo diff $wf $f
   sdiff $wf $f
done

exit 0;
