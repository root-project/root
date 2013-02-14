############################################################################
# CMakeLists.txt file for building ROOT math/mathmore package
############################################################################

ROOT_USE_PACKAGE(math/mathcore)

include_directories(${GSL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
add_definitions(-DUSE_ROOT_ERROR ${GSL_CFLAGS})

set(headers Math/DistFuncMathMore.h Math/SpecFuncMathMore.h Math/PdfFuncMathMore.h
            Math/Polynomial.h Math/Derivator.h Math/Interpolator.h 
            Math/InterpolationTypes.h Math/GSLRootFinder.h Math/GSLRootFinderDeriv.h
            Math/RootFinderAlgorithms.h Math/GSLIntegrator.h Math/GSLMCIntegrator.h 
            Math/MCParameters.h Math/GSLMinimizer1D.h Math/ChebyshevApprox.h 
            Math/Random.h Math/GSLRndmEngines.h Math/KelvinFunctions.h
            Math/GSLMinimizer.h Math/GSLNLSMinimizer.h Math/GSLSimAnMinimizer.h
			Math/GSLMultiRootFinder.h 
            Math/Vavilov.h Math/VavilovAccurate.h Math/VavilovAccuratePdf.h
            Math/VavilovAccurateCdf.h Math/VavilovAccurateQuantile.h Math/VavilovFast.h )
set(linkdefs Math/LinkDef.h Math/LinkDef_Func.h Math/LinkDef_RootFinding.h)

ROOT_GENERATE_DICTIONARY(G__MathMore ${headers} LINKDEF Math/LinkDef.h)
ROOT_GENERATE_ROOTMAP(MathMore LINKDEF ${linkdefs})
ROOT_LINKER_LIBRARY(MathMore *.cxx G__MathMore.cxx LIBRARIES ${GSL_LIBRARIES} DEPENDENCIES MathCore)
ROOT_INSTALL_HEADERS()






