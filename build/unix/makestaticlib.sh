#! /bin/sh

# Script to generate the libRoot.a archive library.
# Called by main Makefile.
#
# Author: Fons Rademakers, 21/01/2001

STATICOBJECTLIST=$1

ROOTALIB=lib/libRoot.a

rm -f $ROOTALIB

objs=`$STATICOBJECTLIST`

echo "Making $ROOTALIB..."
echo ar rv $ROOTALIB cint/cint/main/G__setup.o cint/cint/src/dict/*.o cint/cint/src/config/*.o $objs
ar rv $ROOTALIB cint/cint/main/G__setup.o cint/cint/src/dict/*.o cint/cint/src/config/*.o $objs > /dev/null 2>&1

arstat=$?
if [ $arstat -ne 0 ]; then
   exit $arstat
fi

exit 0
