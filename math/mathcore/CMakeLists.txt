############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

ROOT_USE_PACKAGE(core)
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # Explicit to avoid circular dependencies mathcore <--> hist :-(

set(MATHCORE_HEADERS TRandom.h 
  TRandom1.h TRandom2.h TRandom3.h TVirtualFitter.h TKDTree.h TKDTreeBinning.h 
  Math/IParamFunction.h Math/IFunction.h Math/ParamFunctor.h Math/Functor.h 
  Math/Minimizer.h Math/MinimizerOptions.h Math/IntegratorOptions.h Math/IOptions.h 
  Math/Integrator.h Math/VirtualIntegrator.h Math/AllIntegrationTypes.h Math/AdaptiveIntegratorMultiDim.h 
  Math/IntegratorMultiDim.h Math/Factory.h Math/FitMethodFunction.h Math/GaussIntegrator.h 
  Math/GaussLegendreIntegrator.h Math/RootFinder.h Math/IRootFinderMethod.h Math/RichardsonDerivator.h 
  Math/BrentMethods.h Math/BrentMinimizer1D.h Math/BrentRootFinder.h Math/DistSampler.h 
  Math/DistSamplerOptions.h Math/GoFTest.h Math/SpecFuncMathCore.h Math/DistFuncMathCore.h
)


ROOT_GENERATE_DICTIONARY(G__Math     TComplex.h TMath.h LINKDEF LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__MathCore ${MATHCORE_HEADERS} LINKDEF LinkDef2.h)
ROOT_GENERATE_DICTIONARY(G__MathFit  Fit/*.h LINKDEF LinkDef3.h)

ROOT_GENERATE_ROOTMAP(MathCore LINKDEF LinkDef1.h LinkDef2.h LinkDef3.h LinkDef_Func.h)

add_definitions(-DUSE_ROOT_ERROR )

ROOT_LINKER_LIBRARY(MathCore *.cxx G__Math.cxx G__MathCore.cxx G__MathFit.cxx LIBRARIES ${CMAKE_THREAD_LIBS_INIT} Cint Core)

ROOT_INSTALL_HEADERS()






