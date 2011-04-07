#! /bin/sh

# Script to generate the file include/compiledata.h.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

COMPILEDATA=$1
CXX=$2
CXXOPT=$3
CXXDEBUG=$4
CXXFLAGS=$5
SOFLAGS=$6
LDFLAGS=$7
SOEXT=$8
SYSLIBS=$9
shift
LIBDIR=$9
shift
ROOTLIBS=$9
shift
RINTLIBS=$9
shift
INCDIR=$9
shift
CUSTOMSHARED=$9
shift
CUSTOMEXE=$9
shift
ARCH=$9
shift
ROOTBUILD=$9
shift

if [ "x$ROOTSYS" = "x" ]; then
   ROOTSYS=$PWD
fi

if [ "$(cd $INCDIR && cygpath -u $PWD)" = "$(cd $ROOTSYS/include && cygpath -u $PWD)" ]; then
   INCDIR=%ROOTSYS%/include
fi
if [ "$(cd $LIBDIR && cygpath -u $PWD)" = "$(cd $ROOTSYS/lib && cygpath -u $PWD)" ]; then
   LIBDIR=%ROOTSYS%/lib
fi

CXXDEBUG=`echo $CXXDEBUG|sed 's,-Fdbin/root.pdb,,g'`

HAVEMT=`(unset VS_UNICODE_OUTPUT; mt '/?' >/dev/null 2>&1) && echo 1`
if [ "$HAVEMT" == "1" ]; then
   HAVEMTDLL=' && if EXIST \"$BuildDir\\$LibName.dll.manifest\" ( mt -nologo -manifest \"$BuildDir\\$LibName.dll.manifest\" \"-outputresource:$BuildDir\\$LibName.dll;2\" && del \"$BuildDir\\$LibName.dll.manifest\" )'
   HAVEMTEXE=' && if EXIST \"$ExeName.exe.manifest\" ( mt -nologo -manifest \"$ExeName.exe.manifest\" \"-outputresource:$ExeName.exe;1\" && del \"$ExeName.exe.manifest\" )'
fi


if [ "$CXX" == "./build/win/cl.sh" ] ; then
   # We are using the wrapper dealing with cygwin path, 
   # which is hard coded to use cl.exe, so let's record 
   # information about cl.exe
   CXX=cl.exe
fi

# Determine the compiler version
COMPILERVERS="`$CXX 2>&1 | grep Version | dos2unix`"

rm -f __compiledata

echo "Running $0"
echo "/* This is file is automatically generated */" > __compiledata
echo "#define BUILD_ARCH \"$ARCH\"" >> __compiledata
echo "#define BUILD_NODE \""`uname -a`"\" " >> __compiledata
echo "#define COMPILER \""`type -path $CXX`"\" " >> __compiledata
echo "#define COMPILERVERS \"$COMPILERVERS\"" >> __compiledata
if [ "$CUSTOMSHARED" = "" ]; then 
   echo "#define  MAKESHAREDLIB \"cl \$Opt -nologo -TP -c $CXXFLAGS \$IncludePath  \$SourceFiles -Fo\$ObjectFiles && bindexplib \$LibName \$ObjectFiles > \$BuildDir\\\\\$LibName.def && lib -nologo -MACHINE:IX86 -out:\$BuildDir\\\\\$LibName.lib \$ObjectFiles -def:\$BuildDir\\\\\$LibName.def && link -nologo \$ObjectFiles -DLL $LDFLAGS -out:\$BuildDir\\\\\$LibName.dll \$BuildDir\\\\\$LibName.exp -LIBPATH:%ROOTSYS%\\\\lib  \$LinkedLibs libCore.lib libCint.lib kernel32.lib advapi32.lib user32.lib gdi32.lib comdlg32.lib winspool.lib $HAVEMTDLL \" " >> __compiledata
else
   echo "#define  MAKESHAREDLIB \"$CUSTOMSHARED\"" >> __compiledata
fi

if [ "$CUSTOMEXE" = "" ]; then 
   echo "#define MAKEEXE \"cl -nologo -TP -Iinclude -I../include -c \$Opt $CXXFLAGS \$IncludePath \$SourceFiles && link -opt:ref $LDFLAGS \$ObjectFiles \$LinkedLibs $SYSLIBS -out:\$ExeName $HAVEMTEXE \""  >> __compiledata
else 
   echo "#define MAKEEXE \"$CUSTOMEXE\"" >> __compiledata
fi

echo "#define CXXOPT \"$CXXOPT\"" >> __compiledata
echo "#define CXXDEBUG \"$CXXDEBUG\"" >> __compiledata
echo "#define ROOTBUILD \"$ROOTBUILD\"" >> __compiledata

echo "#define LINKEDLIBS \"-LIBPATH:%ROOTSYS% $ROOTLIBS $RINTLIBS \""  >> __compiledata

echo "#define INCLUDEPATH \"-I$INCDIR\"" >> __compiledata
echo "#define OBJEXT \"obj\" " >> __compiledata
echo "#define SOEXT \"$SOEXT\" " >> __compiledata

(
if [ -r $COMPILEDATA ]; then
   diff __compiledata $COMPILEDATA > /dev/null; status=$?;
   if [ "$status" -ne "0" ]; then
      echo "Changing $COMPILEDATA"
      mv __compiledata $COMPILEDATA;
   else
      rm -f __compiledata; fi
else
   echo "Making $COMPILEDATA"
   mv __compiledata $COMPILEDATA; fi
) 

exit 0

