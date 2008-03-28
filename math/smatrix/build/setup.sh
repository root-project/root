cp -p build/configure.in . 
cp -p build/Makefile.am . 
cp -p build/inc_Makefile.am inc/Makefile.am
cp -p build/inc_Math_Makefile.am inc/Math/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/doc_Makefile.am doc/Makefile.am
cp -p build/autogen . 
cp -p -r build/config . 

./autogen