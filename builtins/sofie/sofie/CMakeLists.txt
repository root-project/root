############################################################################
# CMakeLists.txt file for building the SOFIE library.
#
# SOFIE converts ONNX model files to C++ code that depends only on the C++
# standard library. The library itself has no dependencies besides the C++
# standard library (C++17 is enough).
############################################################################

cmake_minimum_required(VERSION 3.16)
project(SOFIE LANGUAGES CXX)

# The library only needs C++17, but it follows the standard chosen by the user
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

add_library(SOFIE SHARED
  src/ParseArgMax.cxx
  src/ParseBasicBinary.cxx
  src/ParseBasicIs.cxx
  src/ParseBasicNary.cxx
  src/ParseBasicUnary.cxx
  src/ParseBatchNormalization.cxx
  src/ParseCast.cxx
  src/ParseClip.cxx
  src/ParseComparision.cxx
  src/ParseConcat.cxx
  src/ParseConstant.cxx
  src/ParseConv.cxx
  src/ParseConvTranspose.cxx
  src/ParseEinsum.cxx
  src/ParseElu.cxx
  src/ParseErf.cxx
  src/ParseExpand.cxx
  src/ParseEyeLike.cxx
  src/ParseFuseBatchnormRelu.cxx
  src/ParseFuseConvAdd.cxx
  src/ParseFuseConvTransposeAdd.cxx
  src/ParseFuseGemmRelu.cxx
  src/ParseFuseMatMulAdd.cxx
  src/ParseGRU.cxx
  src/ParseGather.cxx
  src/ParseGatherND.cxx
  src/ParseGelu.cxx
  src/ParseGemm.cxx
  src/ParseHardSigmoid.cxx
  src/ParseHardSwish.cxx
  src/ParseIdentity.cxx
  src/ParseIf.cxx
  src/ParseInstanceNormalization.cxx
  src/ParseLSTM.cxx
  src/ParseLayerNormalization.cxx
  src/ParseLeakyRelu.cxx
  src/ParseMatMul.cxx
  src/ParseNonZero.cxx
  src/ParseNot.cxx
  src/ParsePRelu.cxx
  src/ParsePad.cxx
  src/ParsePool.cxx
  src/ParseRNN.cxx
  src/ParseRandom.cxx
  src/ParseRange.cxx
  src/ParseReduce.cxx
  src/ParseRelu.cxx
  src/ParseReshape.cxx
  src/ParseScatterElements.cxx
  src/ParseScatterND.cxx
  src/ParseSelu.cxx
  src/ParseShape.cxx
  src/ParseSigmoid.cxx
  src/ParseSlice.cxx
  src/ParseSoftmax.cxx
  src/ParseSplit.cxx
  src/ParseSwish.cxx
  src/ParseTanh.cxx
  src/ParseThresholdedRelu.cxx
  src/ParseTile.cxx
  src/ParseTopK.cxx
  src/ParseTranspose.cxx
  src/ParseWhere.cxx
  src/RModel.cxx
  src/RModelParser_ONNX.cxx
  src/SOFIE.cxx
  src/common.cxx
  src/common_helpers.cxx
)
add_library(SOFIE::SOFIE ALIAS SOFIE)

# Export the functions of the C interface (see SOFIE/SOFIE.h)
target_compile_definitions(SOFIE PRIVATE SOFIE_BUILDING_LIBRARY)

target_include_directories(SOFIE
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    # onnx.hxx, the self-contained ONNX protobuf reader, is private
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Only the library is installed: it is meant to be loaded at runtime, so its
# headers are not needed by the users
install(TARGETS SOFIE
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
