#! /bin/sh

# Script to generate the file cint/MAKEINFO.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

MAKEINFO=$1
CXX=$2
CC=$3
CPPPREP=$4

rm -f __makeinfo

echo "Running $0"
echo "# This file had been automatically generated" > __makeinfo
echo "# And will be re-generated when running make" >> __makeinfo
echo "# Unless the keyword DEFAULT is replaced by LOCAL" >> __makeinfo
echo "# on the next line, before MAKEINFO" >> __makeinfo
echo "# DEFAULT MAKEINFO" >> __makeinfo
echo "" >> __makeinfo
echo "CC          = $CC" >> __makeinfo
echo "CPP         = $CXX" >> __makeinfo
echo "CPREP       = $CC -E -C" >> __makeinfo
if [ "$CPPPREP" = "" ]; then 
    echo "CPPPREP     = $CXX -E -C" >> __makeinfo
else
    echo "CPPPREP     = $CPPPREP" >> __makeinfo    
fi;
echo "" >> __makeinfo
echo "# source code postfix" >> __makeinfo
echo "CSRCPOST    = .c" >> __makeinfo
echo "CHDRPOST    = .h" >> __makeinfo
echo "CPPSRCPOST  = .cxx" >> __makeinfo
echo "CPPHDRPOST  = .h" >> __makeinfo
echo "OBJPOST     = .o" >> __makeinfo
echo "DLLPOST     = .dl" >> __makeinfo

(
if [ -r $MAKEINFO ]; then
   diff __makeinfo $MAKEINFO > /dev/null; status=$? ;
   if [ "$status" -ne "0" ]; then
      grep " LOCAL MAKEINFO" $MAKEINFO; status=$?;
      if [ "$status" -ne "0" ]; then
         echo "Changing $MAKEINFO"
         mv __makeinfo $MAKEINFO;
      else
         rm -f __makeinfo; fi;
   else
      rm -f __makeinfo; fi;
else
   echo "Making $MAKEINFO"
   mv __makeinfo $MAKEINFO; fi
) 
