# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/genvector package
############################################################################

message(STATUS "Experimental GenVectorX")


include("${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/tools/AdaptiveCpp/adaptivecpp-config.cmake")
export(PACKAGE AdaptiveCpp)

ROOT_LINKER_LIBRARY(GenVectorSYCL
    src/3DConversions.cxx
    src/3DDistances.cxx
    src/AxisAngle.cxx
    src/AxisAngleXother.cxx
    src/BitReproducible.cxx
    src/Boost.cxx
    src/BoostX.cxx
    src/BoostY.cxx
    src/BoostZ.cxx
    src/EulerAngles.cxx
    src/LorentzRotation.cxx
    src/Quaternion.cxx
    src/QuaternionXaxial.cxx
    src/Rotation3D.cxx
    src/Rotation3DxAxial.cxx
    src/RotationZYX.cxx
    src/VectorUtil.cxx
  DEPENDENCIES
    Core
    MathCore
)

add_sycl_to_target(
  TARGET 
    GenVectorSYCL 
  SOURCES 
    src/3DConversions.cxx
    src/3DDistances.cxx
    src/AxisAngle.cxx
    src/AxisAngleXother.cxx
    src/BitReproducible.cxx
    src/Boost.cxx
    src/BoostX.cxx
    src/BoostY.cxx
    src/BoostZ.cxx
    src/EulerAngles.cxx
    src/LorentzRotation.cxx
    src/Quaternion.cxx
    src/QuaternionXaxial.cxx
    src/Rotation3D.cxx
    src/Rotation3DxAxial.cxx
    src/RotationZYX.cxx
    src/VectorUtil.cxx
)

set(ACPP_EXPORTS acpp-rt acpp-common)

foreach(target IN LISTS ACPP_EXPORTS)
   install(TARGETS ${target} EXPORT ROOTExports)
endforeach()

target_include_directories(GenVectorSYCL SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/interpreter/llvm-project/llvm/tools/AdaptiveCpp/include/AdaptiveCpp>
  $<INSTALL_INTERFACE:include/AdaptiveCpp>)

target_link_libraries(GenVectorSYCL INTERFACE acpp-rt)
target_compile_options(GenVectorSYCL PUBLIC ${SYCL_COMPILER_FLAGS})
target_compile_definitions(GenVectorSYCL PUBLIC ROOT_MATH_SYCL)

ROOT_ADD_TEST_SUBDIRECTORY(test)

ROOT_INSTALL_HEADERS()