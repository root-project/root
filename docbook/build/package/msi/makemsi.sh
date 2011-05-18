#!/bin/sh

#
# Creates a msi installer package, using WiX (http://wix.sf.net)
#
# Axel, 2006-05-05

# USAGE: makemsi outputfile.msi -T filelist.txt
# will create a MSI file for files in filelist.txt

if ! which candle > /dev/null 2>&1; then
   echo ""
   echo '   WiX not found!'
   echo ""
   echo 'Please download the VERSION 3 wix binaries from http://wix.sf.net,'
   echo 'extract them, and put them into your $PATH so I can find them.'
   exit 1
fi

MSIFILE=$1
[ "$MSIFILE" = "" ] && MSIFILE=root.msi

MSIDIR=`dirname $MSIFILE`
[ "$MSIDIR" = "" ] && MSIDIR=$PWD
[ -d $MSIDIR ] || mkdir -p $MSIDIR

WIXDIR=`which light | sed 's,^\(.*/\)[^/].*$,\1,'`
ROOTSYS=$PWD

ROOTXMS=$MSIDIR/ROOT.xms
ROOTXMSO=$MSIDIR/ROOT.wixobj

shift
echo `cygpath -m $ROOTXMS` $* | xargs build/package/msi/makemsi.exe || exit 1

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
$CMD || exit 1

CMD="light -nologo -out `cygpath -w $MSIFILE` `cygpath -w $ROOTXMSO` `cygpath -w $WIXDIR/lib/wixui_mondo.wixlib` -loc `cygpath -w $ROOTSYS/build/package/msi/Language_en-us.wxl`"
echo $CMD
$CMD || exit 1
echo ""
rm $ROOTXMSO $ROOTXMS
chmod a+rx $MSIFILE
echo "Done."
