#! /bin/sh

#dt_MakeFiles.sh
#root -b -q 'dt_MakeRef.C("Event.new.split9.root");'
if [ "x`which gmake 2>/dev/null | awk '{if ($1~/gmake/) print $1;}'`" != "x" ]
then
   gmake -f dt_Makefile drawtest
else
   make -f dt_Makefile drawtest
fi
./dt_RunDrawTest.sh
