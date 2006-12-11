cp -p build/configure.in . 
cp -p build/Makefile.am . 
cp -p build/src_Makefile.am src/Makefile.am
cp -p build/inc_Makefile.am inc/Makefile.am
cp -p build/inc_Math_Makefile.am inc/Math/Makefile.am
cp -p build/test_Makefile.am test/Makefile.am
cp -p build/doc_Makefile.am doc/Makefile.am
cp -p build/autogen . 
cp -p -r build/config . 

# add files from mahcore
cp ../mathcore/inc/Math/IFunction.h inc/Math/.
cp ../mathcore/inc/Math/IFunctionfwd.h inc/Math/.
cp ../mathcore/inc/Math/IParamFunction.h inc/Math/.
cp ../mathcore/inc/Math/IParamFunctionfwd.h inc/Math/.
cp ../mathcore/inc/Math/Functor.h inc/Math/.
cp ../mathcore/inc/Math/Util.h inc/Math/.
cp ../mathcore/inc/Math/WrappedFunction.h inc/Math/.
cp ../mathcore/inc/Math/WrappedParamFunction.h inc/Math/.

./autogen