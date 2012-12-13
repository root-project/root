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
echo ar rv $ROOTALIB core/textinput/src/textinput/*.o $objs
ar rv $ROOTALIB core/textinput/src/textinput/*.o $objs > /dev/null 2>&1

arstat=$?
if [ $arstat -ne 0 ]; then
   exit $arstat
fi

exit 0
