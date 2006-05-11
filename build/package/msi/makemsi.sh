#!/bin/sh

#
# Creates a msi installer package, using WiX (http://wix.sf.net)
# Package is created in ../, and called root_v9.87.65win32gdk.msi
#
# Axel, 2006-05-05
#
# Parameters
# 1: [optional] output dir for msi file (/tmp/msi/ otherwise)
# 2: if it is "reuse", reuse the xms source file generated form a previous run.
# Note that for $2 to be "reuse" you'll have to specify $1!

if ! which candle > /dev/null 2>&1; then
   echo ""
   echo '   WiX not found!'
   echo ""
   echo 'Please download the VERSION 3 wix binaries from http://wix.sf.net,'
   echo 'extract them, and put them into your $PATH so I can find them.'
   exit 1
fi

MSIDIR=$1
[ "$MSIDIR" == "" ] && MSIDIR=/tmp/msi
[ -d $MSIDIR ] || mkdir -p $MSIDIR

WIXDIR=`which light | sed 's,^\(.*/\)[^/].*$,\1,'`
[ "$ROOTSYS" == "" ] && ROOTSYS=$PWD

VERSIONNUMBER=`cat $ROOTSYS/build/version_number | sed 's,/,.,g'`
DEBUG=`$ROOTSYS/bin/root-config --config | grep 'build=debug' | sed 's,^.*--build=.*$,.debug,'`
MSIFILE=${MSIDIR}/root_v${VERSIONNUMBER}.win32gdk${DEBUG}.msi
ROOTXMS=$MSIDIR/ROOT.xms

[ "$2" != "reuse" ] && $ROOTSYS/build/package/msi/makemsi.exe `cygpath -m $MSIDIR` || exit 1

# fix WiX UI problem...
if [ ! -d $WIXDIR/lib/Bitmaps -o ! -f $WIXDIR/lib/Bitmaps/dlgbmp.bmp ]; then
   mkdir $WIXDIR/lib/Bitmaps
   mv $WIXDIR/lib/*.ico $WIXDIR/lib/Bitmaps/
   mv $WIXDIR/lib/*.bmp $WIXDIR/lib/Bitmaps/
fi

# stupid license file cannot be anywhere else but in $WIXDIR, or we have to re-generate the UI...
[ -f $WIXDIR/lib/License.rtf.orig ] || mv -f $WIXDIR/lib/License.rtf $WIXDIR/lib/License.rtf.orig 
cp $ROOTSYS/build/package/msi/License.rtf $WIXDIR/lib/License.rtf

# now compile!
echo "Compiling and linking - this will take a while..."
CMD="candle -nologo -sw1044 `cygpath -w $ROOTXMS` -out `cygpath -w $MSIDIR/root.wixobj`"
echo $CMD
$CMD || exit

CMD="light -nologo -out `cygpath -w $MSIFILE` `cygpath -w $MSIDIR/root.wixobj` `cygpath -w $WIXDIR/lib/wixui_mondo.wixlib` -loc `cygpath -w $ROOTSYS/build/package/msi/Language_en-us.wxl`"
echo $CMD
$CMD || exit
echo ""
[ "$2" != "reuse" ] && rm $MSIDIR/root.wixobj $MSIDIR/ROOT.xms
echo "Done."
