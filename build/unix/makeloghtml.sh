#! /bin/sh

ROOT=bin/root.exe
ROOTCONFIG=bin/root-config

VERS=`$ROOTCONFIG --prefix=. --version`

echo ""
echo "Generating hyperized version of README/ChangeLog in directory htmldoc/..."
echo ""

$ROOT -b <<makedoc
    THtml html;
    html.Convert("README/ChangeLog", "ROOT Version $VERS Release Notes");
    .q
makedoc

exit 0
