#! /bin/sh

# Script to generate the file cint/MAKEINFO.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

MAKEINFO=$1
CXX=$2
CC=$3

rm -f __makeinfo

echo "Making $MAKEINFO"
echo "# This is file had been automatically generated" > __makeinfo
echo "# And will be re-generated when running make" >> __makeinfo
echo "# Unless the keyword DEFAULT is replaced by LOCAL" >> __makeinfo
echo "# on the next line, before MAKEINFO" >> __makeinfo
echo "# DEFAULT MAKEINFO" >> __makeinfo
echo "" >> __makeinfo
echo "CC          = $CC" >> __makeinfo
echo "CPP         = $CXX" >> __makeinfo
echo "CPREP       = $CC -E -C" >> __makeinfo
echo "CPPPREP     = $CXX -E -C" >> __makeinfo
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
   diff __makeinfo $MAKEINFO; status=$?;
   if [ "$status" -ne "0" ]; then
      grep " LOCAL MAKEINFO" $MAKEINFO; status=$?;
      if [ "$status" -ne "0" ]; then
         mv __makeinfo $MAKEINFO;
      else
         touch $MAKEINFO; rm -f __makeinfo; fi;
   else
      touch $MAKEINFO; rm -f __makeinfo; fi;
else
   mv __makeinfo $MAKEINFO; fi
) > /dev/null
