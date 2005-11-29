cp -p build/configure.in . 
cp -p build/Makefile.am . 
cp -p build/src_Makefile.am src/Makefile.am
cp -p build/inc_Makefile.am inc/Makefile.am
cp -p build/inc_Minuit2_Makefile.am inc/Minuit2/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/test_MnSim_Makefile.am test/MnSim/Makefile.am
cp -p build/test_MnTutorial_Makefile.am test/MnTutorial/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/doc_Makefile.am doc/Makefile.am
cp -p build/autogen . 
cp -p -r build/config . 

./autogen