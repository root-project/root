#! /bin/sh

dir=`pwd`
ROOT=$dir/bin/root
ROOTCONFIG=$dir/bin/root-config

VERS=`$ROOTCONFIG --prefix=. --version`

echo ""
echo "Generating hyperized version of README/ChangeLog in directory htmldoc/..."
echo ""

$ROOT -b -l <<makedoc
    THtml html;
    html.Convert("README/ChangeLog", "ROOT Version $VERS Release Notes");
    .q
makedoc

exit 0
