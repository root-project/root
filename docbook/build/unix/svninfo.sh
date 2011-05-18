#!/usr/bin/env bash

# Store info about which svn branch, what revision and at what date/time
# we executed make.

dir=
dotsvn=".svn"
svn=svn
if [ $# = 1 ]; then
   if [ -x /bin/cygpath ]; then
      dir=`cygpath -u $1`
      svn=/usr/bin/svn
   else
      dir=$1
   fi
   dotsvn="$dir/.svn"
fi

# if we don't see the .svn directory, just return
if test ! -d $dotsvn; then
   exit 0;
fi

OUT=etc/svninfo.txt

INFO=`$svn info $dir | awk '/Last Changed Rev:/ { print $4 } /URL:/ { print $2 }'`

HTTP="http://root.cern.ch/svn/root/"
HTTPS="https://root.cern.ch/svn/root/"

for i in $INFO; do
   if test ${i:0:4} = "http"; then
      if test ${i:0:5} = "https"; then
         echo ${i#${HTTPS}} > $OUT
      else
         echo ${i#${HTTP}} > $OUT
      fi
   else
      echo $i >> $OUT
      echo $i
   fi
done

date "+%b %d %Y, %H:%M:%S" >> $OUT

exit 0
