#! /bin/sh

ROOT=bin/root.exe
OUT=etc/system.plugins-ios
DATE=`date`

echo ""
echo "Generating etc/system.plugins-ios"
echo ""

$ROOT -l <<makeplugins
    gPluginMgr->WritePluginRecords("plugins-ios");
    .q
makeplugins

echo "# This file has been generated using:" >  $OUT
echo "#    make plugins-ios"                 >> $OUT
echo "# On $DATE"                            >> $OUT
echo "# DON'T MAKE CHANGES AS THEY WILL GET LOST NEXT TIME THE FILE IS GENERATED" >> $OUT
cat plugins-ios >> $OUT
rm -f plugins-ios

exit 0
