#! /bin/sh

# Script to generate the file include/compiledata.h.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

COMPILEDATA=$1
CXX=$2
OPT=$3
CXXFLAGS=$4
SOFLAGS=$5
LDFLAGS=$6
SOEXT=$7
SYSLIBS=$8
LIBDIR=$9
shift
ROOTLIBS=$9
shift
RINTLIBS=$9
shift
INCDIR=$9

if [ "$INCDIR" = "$ROOTSYS/include" ]; then
   INCDIR=\$ROOTSYS/include
fi
if [ "$LIBDIR" = "$ROOTSYS/lib" ]; then
   LIBDIR=\$ROOTSYS/lib
fi

rm -f __compiledata

echo "Making $COMPILEDATA"
echo "/* This is file is automatically generated */" > __compiledata
echo "#define MAKESHAREDLIB  \"$CXX -c $OPT $CXXFLAGS \$IncludePath \$SourceFiles ; $CXX \$ObjectFiles $SOFLAGS $LDFLAGS -o \$SharedLib\"" >> __compiledata
echo "#define MAKEEXE \"$CXX -c $OPT $CXXFLAGS \$IncludePath \$SourceFiles; $CXX \$ObjectFiles $LDFLAGS -o \$ExeName \$LinkedLibs $SYSLIBS\""  >> __compiledata
echo "#define LINKEDLIBS \"-L$LIBDIR $ROOTLIBS $RINTLIBS \""  >> __compiledata
echo "#define INCLUDEPATH \"-I$INCDIR\"" >> __compiledata
echo "#define OBJEXT \"o\" " >> __compiledata
echo "#define SOEXT \"$SOEXT\" " >> __compiledata

(
if [ -r $COMPILEDATA ]; then
   diff __compiledata $COMPILEDATA; status=$?;
   if [ "$status" -ne "0" ]; then
      mv __compiledata $COMPILEDATA;
   else
      touch $COMPILEDATA; rm -f __compiledata; fi;
else
   mv __compiledata $COMPILEDATA; fi
) > /dev/null

exit 0

