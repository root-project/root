############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(MATHCORE_HEADERS TRandom.h
  TRandom1.h TRandom2.h TRandom3.h TKDTree.h TKDTreeBinning.h TStatistic.h
  Math/Error.h Math/IParamFunction.h Math/IFunction.h Math/ParamFunctor.h Math/Functor.h
  Math/Minimizer.h Math/MinimizerOptions.h Math/IntegratorOptions.h Math/IOptions.h Math/GenAlgoOptions.h
  Math/BasicMinimizer.h Math/MinimTransformFunction.h Math/MinimTransformVariable.h Math/IFunctionfwd.h
  Math/Integrator.h Math/VirtualIntegrator.h Math/AllIntegrationTypes.h Math/AdaptiveIntegratorMultiDim.h
  Math/IntegratorMultiDim.h Math/Factory.h Math/FitMethodFunction.h Math/GaussIntegrator.h
  Math/GaussLegendreIntegrator.h Math/RootFinder.h Math/IRootFinderMethod.h Math/RichardsonDerivator.h
  Math/BrentMethods.h Math/BrentMinimizer1D.h Math/BrentRootFinder.h Math/DistSampler.h
  Math/DistSamplerOptions.h Math/GoFTest.h Math/SpecFuncMathCore.h Math/DistFuncMathCore.h
  Math/ChebyshevPol.h Math/KDTree.h Math/TDataPoint.h Math/TDataPointN.h Math/Delaunay2D.h
  Math/Random.h Math/TRandomEngine.h Math/RandomFunctions.h Math/StdEngine.h Math/Math.h
  Math/MersenneTwisterEngine.h Math/MixMaxEngine.h   TRandomGen.h Math/LCGEngine.h
  Math/Types.h Math/Util.h Math/WrappedFunction.h Math/IMinimizer1D.h Math/IParamFunctionfwd.h
  Math/MinimizerVariableTransformation.h Math/MultiDimParamFunctionAdapter.h Math/OneDimFunctionAdapter.h
  Math/PdfFunc.h Math/PdfFuncMathCore.h Math/ProbFunc.h Math/ProbFuncMathCore.h Math/QuantFunc.h
  Math/QuantFuncMathCore.h Math/WrappedParamFunction.h
  )

if(runtime_cxxmodules)
  # Adding Math/Math.h to the MathCore module makes the generation of GenVector
  # dictionary hand due to a bug in rootcling/clang. The redeclaration chain of
  # 'abs' shadow declaration (from stl math.h) is broken.
  # FIXME: Revise after a llvm upgrade or reproduce it outside rootcling.
  list(REMOVE_ITEM MATHCORE_HEADERS "Math/Math.h")
endif()

if(veccore)
  set(MATHCORE_LIBRARIES ${VecCore_LIBRARIES})
  set(MATHCORE_BUILTINS VECCORE)
endif()

ROOT_ADD_C_FLAG(_flags -Wno-strict-overflow)  # Avoid what it seems a compiler false positive warning
ROOT_ADD_C_FLAG(_flags -Wno-maybe-uninitialized)  # Avoid what it seems a compiler false positive warning
ROOT_ADD_C_FLAG(_flags -Wno-parentheses-equality)

set_source_files_properties(src/triangle.c COMPILE_FLAGS "${_flags}")

if(imt)
  set(MATHCORE_DEPENDENCIES Imt)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(MathCore HEADERS TComplex.h TMath.h ${MATHCORE_HEADERS} Fit/*.h
                              SOURCES *.cxx *.c
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${MATHCORE_LIBRARIES}
                              DEPENDENCIES Core ${MATHCORE_DEPENDENCIES}
                              BUILTINS ${MATHCORE_BUILTINS})

if(veccore)
  target_compile_definitions(MathCore INTERFACE ${VecCore_DEFINITIONS})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
