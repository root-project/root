# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT RooFit package
# @author Pere Mato, CERN
############################################################################

add_library(RooFit)
target_sources(RooFit
PRIVATE
  src/Roo2DKeysPdf.cxx
  src/RooArgusBG.cxx
  src/RooBCPEffDecay.cxx
  src/RooBCPGenDecay.cxx
  src/RooBDecay.cxx
  src/RooBMixDecay.cxx
  src/RooBernstein.cxx
  src/RooBifurGauss.cxx
  src/RooBlindTools.cxx
  src/RooBreitWigner.cxx
  src/RooBukinPdf.cxx
  src/RooCBShape.cxx
  src/RooCFunction1Binding.cxx
  src/RooCFunction2Binding.cxx
  src/RooCFunction3Binding.cxx
  src/RooCFunction4Binding.cxx
  src/RooChebychev.cxx
  src/RooChi2MCSModule.cxx
  src/RooChiSquarePdf.cxx
  src/RooCrystalBall.cxx
  src/RooDecay.cxx
  src/RooDstD0BG.cxx
  src/RooExponential.cxx
  src/RooFunctor1DBinding.cxx
  src/RooFunctorBinding.cxx
  src/RooGExpModel.cxx
  src/RooGamma.cxx
  src/RooGaussExpTails.cxx
  src/RooGaussModel.cxx
  src/RooGaussian.cxx
  src/RooHistConstraint.cxx
  src/RooIntegralMorph.cxx
  src/RooJeffreysPrior.cxx
  src/RooJohnson.cxx
  src/RooKeysPdf.cxx
  src/RooLagrangianMorphFunc.cxx
  src/RooLandau.cxx
  src/RooLegacyExpPoly.cxx
  src/RooLognormal.cxx
  src/RooMathCoreReg.cxx
  src/RooMomentMorph.cxx
  src/RooMomentMorphFunc.cxx
  src/RooMomentMorphFuncND.cxx
  src/RooMultiBinomial.cxx
  src/RooNDKeysPdf.cxx
  src/RooNonCPEigenDecay.cxx
  src/RooNovosibirsk.cxx
  src/RooONNXFunc.cxx
  src/RooParamHistFunc.cxx
  src/RooParametricStepFunction.cxx
  src/RooPoisson.cxx
  src/RooPolynomial.cxx
  src/RooPowerSum.cxx
  src/RooSpline.cxx
  src/RooStepFunction.cxx
  src/RooStudentT.cxx
  src/RooTFnBinding.cxx
  src/RooTFnPdfBinding.cxx
  src/RooTMathReg.cxx
  src/RooUnblindCPAsymVar.cxx
  src/RooUnblindOffset.cxx
  src/RooUnblindPrecision.cxx
  src/RooUnblindUniform.cxx
  src/RooUniform.cxx
  src/RooVoigtian.cxx
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    src/RooFit/Detail/Algorithms.h
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/Roo2DKeysPdf.h
    inc/RooArgusBG.h
    inc/RooBCPEffDecay.h
    inc/RooBCPGenDecay.h
    inc/RooBDecay.h
    inc/RooBMixDecay.h
    inc/RooBernstein.h
    inc/RooBifurGauss.h
    inc/RooBlindTools.h
    inc/RooBreitWigner.h
    inc/RooBukinPdf.h
    inc/RooCBShape.h
    inc/RooCFunction1Binding.h
    inc/RooCFunction2Binding.h
    inc/RooCFunction3Binding.h
    inc/RooCFunction4Binding.h
    inc/RooChebychev.h
    inc/RooChi2MCSModule.h
    inc/RooChiSquarePdf.h
    inc/RooCrystalBall.h
    inc/RooDecay.h
    inc/RooDstD0BG.h
    inc/RooExponential.h
    inc/RooFit/Detail/CFunctionBinding.h
    inc/RooFunctor1DBinding.h
    inc/RooFunctorBinding.h
    inc/RooGExpModel.h
    inc/RooGamma.h
    inc/RooGaussExpTails.h
    inc/RooGaussModel.h
    inc/RooGaussian.h
    inc/RooHistConstraint.h
    inc/RooIntegralMorph.h
    inc/RooJeffreysPrior.h
    inc/RooJohnson.h
    inc/RooKeysPdf.h
    inc/RooLagrangianMorphFunc.h
    inc/RooLandau.h
    inc/RooLegacyExpPoly.h
    inc/RooLognormal.h
    inc/RooMathCoreReg.h
    inc/RooMomentMorph.h
    inc/RooMomentMorphFunc.h
    inc/RooMomentMorphFuncND.h
    inc/RooMultiBinomial.h
    inc/RooNDKeysPdf.h
    inc/RooNonCPEigenDecay.h
    inc/RooNovosibirsk.h
    inc/RooONNXFunc.h
    inc/RooParamHistFunc.h
    inc/RooParametricStepFunction.h
    inc/RooPoisson.h
    inc/RooPolynomial.h
    inc/RooPowerSum.h
    inc/RooPyBind.h
    inc/RooSpline.h
    inc/RooStepFunction.h
    inc/RooStudentT.h
    inc/RooTFnBinding.h
    inc/RooTFnPdfBinding.h
    inc/RooTMathReg.h
    inc/RooUnblindCPAsymVar.h
    inc/RooUnblindOffset.h
    inc/RooUnblindPrecision.h
    inc/RooUnblindUniform.h
    inc/RooUniform.h
    inc/RooVoigtian.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef1.h
)

target_link_libraries(RooFit
PUBLIC
    Core
    RooFitCore
    Tree
    RIO
    Matrix
    MathCore
)

# For recent clang, this can facilitate auto-vectorisation.
# In RooFit, the errno side effect is not needed, anyway:
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options(RooFit PUBLIC -fno-math-errno)
endif()

# workaround to include privately headers that are not public, from a "friend" target
get_target_property(RooFitCore_Friend_Dirs RooFitCore HEADER_DIRS_friend_header_files)
if(RooFitCore_Friend_Dirs)
  target_include_directories(RooFit PRIVATE ${RooFitCore_Friend_Dirs})
endif()
get_target_property(RooBatchComputeGeneric_Friend_Dirs RooBatchCompute_GENERIC HEADER_DIRS_friend_header_files)
if(RooBatchComputeGeneric_Friend_Dirs)
  target_include_directories(RooFit PRIVATE ${RooBatchComputeGeneric_Friend_Dirs})
endif()
get_target_property(RooBatchCompute_Friend_Dirs RooBatchCompute HEADER_DIRS_friend_header_files)
if(RooBatchCompute_Friend_Dirs)
  target_include_directories(RooFit PRIVATE ${RooBatchCompute_Friend_Dirs})
endif()

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_GENERATE_TARGET_DICTIONARY(RooFit "-writeEmptyRootPCM" ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(RooFit)

ROOT_ADD_TEST_SUBDIRECTORY(test)
