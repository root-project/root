#! /bin/sh

# Script to import new version of CINT in the ROOT CVS tree.
# Called by main Makefile. Assumes original CINT distribution
# is in $HOME/cint.
#
# Author: Fons Rademakers, 29/2/2000

if [ ! -d $HOME/cint ]; then
   echo "Cannot find original CINT directory: $HOME/cint"
   exit 1;
fi

# original locations
ORG=$HOME/cint
INCO=$ORG/src
SRCO=$ORG/src
INCLO=$ORG/include
LIBO=$ORG/lib
MAINO=$ORG/main
STLO=$ORG/stl
TOOLO=$ORG/tool

# assembly directories
ASM=/tmp/cint
INCT=$ASM/inc
SRCT=$ASM/src
INCLT=$ASM/include
LIBT=$ASM/lib
MAINT=$ASM/main
STLT=$ASM/stl
TOOLT=$ASM/tool

# final destinations
DST=cint
INC=$DST/inc
SRC=$DST/src
INCL=$DST/include
LIB=$DST/lib
MAIN=$DST/main
STL=$DST/stl
TOOL=$DST/tool

# files containing list of new and old files
NEWF=/tmp/cintasm
OLDF=/tmp/cintcvs
ADDED=/tmp/cintadded
REMOVED=/tmp/cintremoved

# create assembly locations
rm -rf $ASM
mkdir -p $INCT $SRCT $INCLT $LIBT $MAINT $STLT $TOOLT

# copy source to assembly area
cp $ORG/*.h $INCT/
cp $INCO/*.h $INCT/
cp $ORG/platform/aixdlfcn/aixdlfcn.h $INCT/

cp $SRCO/*.c $SRCT/
cp $SRCO/*.cxx $SRCT/
cp $ORG/platform/aixdlfcn/dlfcn.c $SRCT/

tar cf - -C $ORG --exclude lib/WildCard --exclude lib/cintocx \
   --exclude lib/wintcldl --exclude lib/wintcldl83 \
   include lib main stl tool | (cd $ASM; tar xf -)

rm -f $SRCT/dmystrm.c
rm -f $SRCT/dmystrct.c
rm -f $SRCT/dmyinit.c
rm -f $INCLT/done
rm -f $INCLT/error
rm -f $INCLT/iosenum.*
rm -f $INCLT/systypes.h
rm -f $INCLT/sys/types.h

# copy man pages directly to man directory
cp $ORG/doc/man1/cint.1 man/man1/
cp $ORG/doc/man1/makecint.1 man/man1/

# compare files in assembly area with the ones in the CVS area

# make a sorted list of all files in the assembly area
find /tmp/cint -type f -print | sort | sed -e "s@/tmp/@@" > $NEWF
find $INC $INCL $LIB $MAIN $SRC $STL $TOOL -path '*/CVS' -prune -o \
     -type f ! -name *.d ! -name *.o -print | sort > $OLDF

comm -23 $OLDF $NEWF > $REMOVED
comm -13 $OLDF $NEWF > $ADDED

echo "Added files (add in CVS):"
cat $ADDED
echo ""
echo "Removed files (remove from CVS):"
cat $REMOVED

# move new files in place
tar cf - -C $ASM inc include lib main src stl tool | (cd $DST; tar xf -)

# cleanup
#rm -rf $ASM $OLDF $NEWF $ADDED $REMOVED
rm -rf $ASM $OLDF $NEWF

exit 0
