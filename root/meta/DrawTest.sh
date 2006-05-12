#! /bin/sh

#dt_MakeFiles.sh
#root -b -q 'dt_MakeRef.C("Event.new.split9.root");'e
MAKEFLAGS=`echo "$MAKEFLAGS" | sed -e 's/V[A-Z]*=[^\w]*//' `
export MAKEFLAGS
unset V
unset VERBOSE

if [ "x`which gmake 2>/dev/null | awk '{if ($1~/gmake/) print $1;}'`" != "x" ]
then
   gmake -f dt_Makefile drawtest -j 1
else
   make -f dt_Makefile drawtest -j 1
fi
./dt_RunDrawTest.sh
