#! /bin/sh

#dt_MakeFiles.sh
#root -b -q 'dt_MakeRef.C("Event.new.split9.root");'
gmake -f dt_Makefile drawtest
dt_RunDrawTest.sh


