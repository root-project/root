cp -p build/configure.in . 
cp -p build/Makefile.am . 
cp -p build/src_Makefile.am src/Makefile.am
cp -p build/inc_Makefile.am inc/Makefile.am
cp -p build/inc_Math_Makefile.am inc/Math/Makefile.am
cp -p build/inc_Math_GenVector_Makefile.am inc/Math/GenVector/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/doc_Makefile.am doc/Makefile.am
cp -p build/autogen . 
cp -p -r build/config . 
#extra files needed 
cp -p ../mathcore/inc/Math/Math.h inc/Math/.

#make dummy file RConfigure.h
#echo "///Dummy file to simulate ROOT configure file
##ifndef ROOT_RConfigure
##define ROOT_RConfigure
##endif" > RConfigure.h
#cp RConfigure.h inc/.

./autogen