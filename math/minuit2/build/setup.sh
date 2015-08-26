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

# add extra files from mathcore
mkdir inc/Math
cp -p ../mathcore/inc/Math/Functor.h inc/Math/.
cp -p ../mathcore/inc/Math/IFunction.h inc/Math/.
cp -p ../mathcore/inc/Math/IFunctionfwd.h inc/Math/.
cp -p ../mathcore/inc/Math/Minimizer.h inc/Math/.
cp -p ../mathcore/inc/Math/FitMethodFunction.h inc/Math/.
cp -p ../mathcore/inc/Math/WrappedFunction.h inc/Math/.
cp -p ../mathcore/inc/Math/MinimizerOptions.h inc/Math/.
cp -p ../mathcore/inc/Math/IOptions.h inc/Math/.
cp -p ../mathcore/inc/Math/Error.h inc/Math/.
cp -p ../mathcore/inc/Math/GenAlgoOptions.h inc/Math/.
cp -p ../mathcore/src/MinimizerOptions.cxx src/.
cp -p ../mathcore/src/GenAlgoOptions.cxx src/.

#hack Minimizer options to define a macro
sed -i '1s/^/#define MATH_NO_PLUGIN_MANAGER\n/' src/MinimizerOptions.cxx 

cp -p build/inc_Math_Makefile.am inc/Math/Makefile.am

./autogen