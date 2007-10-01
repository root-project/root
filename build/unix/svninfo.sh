#! /bin/sh

# Store info about which svn branch, what revision and at what date/time
# we executed make.

OUT=etc/svninfo.txt

INFO=`svn info | awk '/Revision:/ { print $2 } /URL:/ { print $2 }'`

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
      echo $i | tee -a $OUT
   fi
done

date "+%b %d %Y, %H:%M:%S" >> $OUT

exit 0
