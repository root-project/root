#!/bin/sh
#
#
#

CXXFLAGS='-O2 -pipe -Wall -W -Woverloaded-virtual -fPIC -pthread'
LDFLAGS='-shared -Wl,-soname,lib${pkgname}.so -O2'

run()
{
# -- Some basic sanity checks
if [ ! -d roofitcore ] ; then
   echo "$0: ERROR script must be run from $ROOTSYS/roofit directory"
   exit 1 ;
fi

if [ "$ROOTSYS" = "" ] ; then
   echo "$0: ERROR \$ROOTSYS is not set"
   exit 2 ;
fi

if [ ! -d $ROOTSYS ] ; then
   echo "$0: ERROR \$ROOTSYS is not accessible"
   exit 3 ;
fi


# -- Clean build area
rm -rf build/
mkdir -p build/inc
mkdir -p build/inc/RooStats
mkdir -p build/lib

# -- Build roofitcore,roofit and roostats
build_library RooFitCore
build_library RooFit
build_library RooStats
}




build_library()
{
# -------------------------------------------
# -  Method to build one library            -
# -------------------------------------------
pkgname=$1
dirname=`echo $pkgname | gawk '{print tolower($1)}'`

# Clean temp area
basebuilddir=`pwd`
rm -rf $basebuilddir/build/tmp
mkdir -p $basebuilddir/build/tmp

# Make list of .h .cxx and linkdef files
HHLIST=`ls -1 ${dirname}/inc/*.h | grep -v LinkDef`
LDLIST=`ls -1 ${dirname}/inc/LinkDef*.h`
CCLIST=`ls -1 ${dirname}/src/*.cxx`

# Ugly hack to handle different include scheme for RooStats
# Set also the library link list for each case
LDLIBS="-Lbuild/lib -L$ROOTSYS/lib"
case $pkgname in
  RooFitCore)
    LDLIBS="$LDLIBS -lFoam -lHist -lGraf -lMatrix -lTree -lMinuit -lRIO -lMathCore -lCore -lCint"
    pkginc=''
  ;;
  RooFit)
    pkginc=''
    LDLIBS="$LDLIBS -lRooFitCore -lTree -lRIO -lMatrix -lMathCore -lCore -lCint"
  ;;
  RooStats)
    pkginc='RooStats/'
    LDLIBS="$LDLIBS -lRooFitCore -lRooFit -lTree -lRIO -lMatrix -lHist -lMathCore -lGraf -lGpad -lCore -lCint"
  ;;
esac

# Copy hh files to build/inc
echo "copying header files for $dirname"
for file in $HHLIST
do
  cp $file $basebuilddir/build/inc/$pkginc
done

# Generate dictionaries
echo "generating cint dictionaries for $dirname"
for file in $LDLIST
do
   NUMBER=`basename $file | gawk '{sub("LinkDef","") ; sub("[.]h","") ; print $1}'`
   TMPFILE=G__${pkgname}${NUMBER}.cxx
   ( cd $basebuilddir/build/inc ; ${ROOTSYS}/bin/rootcint -cint -f $basebuilddir/build/tmp/$TMPFILE -c -I$basebuilddir/build/inc `ls -1 ${pkginc}*.h` $basebuilddir/$file )
   CCLIST="$CCLIST build/tmp/$TMPFILE"
done

# Compile cxx files
for file in $CCLIST
do
    echo "compiling $file"
    OBJ=`basename $file | gawk '{sub("cxx$","o") ; print $1}'`
    echo g++ ${CXXFLAGS} -Ibuild/inc -I${ROOTSYS}/include -o build/tmp/$OBJ -c $file
    g++ ${CXXFLAGS} -Ibuild/inc -I${ROOTSYS}/include -o build/tmp/$OBJ -c $file
done

# Make shared library
echo "building shared library for $dirname"
echo g++ $LDFLAGS $LDLIBS -o build/lib/lib${pkgname}.so build/tmp/*.o
g++ $LDFLAGS $LDLIBS -o build/lib/lib${pkgname}.so build/tmp/*.o

#rm -rf build/tmp
}

run

