cp -p build/configure.in . 
cp -p build/Makefile.am . 
cp -p build/src_Makefile.am src/Makefile.am
cp -p build/inc_Makefile.am inc/Makefile.am
cp -p build/inc_Math_Makefile.am inc/Math/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/doc_Makefile.am doc/Makefile.am
cp -p build/autogen . 
cp -p -r build/config . 

# add interface files from mathcore
cp ../mathcore/inc/Math/*.h inc/Math/.
mkdir inc/Math/GenVector
cp ../mathcore/src/*.cxx src/.
cp ../mathcore/src/*.h    src/.
cp ../mathcore/inc/Math/GenVector/*.h inc/Math/GenVector/.
cp ../mathcore/build/inc_Math_GenVector_Makefile.am inc/Math/GenVector/Makefile.am


#make file RConfigure.h required to know if mathmore is there in mathcore files 
echo "///Dummy file to simulate ROOT configure file
#ifndef ROOT_RConfigure
#define ROOT_RConfigure
#define R__HAS_MATHMORE
#define MATH_NO_PLUGIN_MANAGER
#endif" > RConfigure.h

cp RConfigure.h inc/.

./autogen

