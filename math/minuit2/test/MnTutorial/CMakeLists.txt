# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.


add_minuit2_test(Quad1F Quad1FMain.cxx Quad1F.h)

add_minuit2_test(Quad4F Quad4FMain.cxx Quad4F.h)

add_minuit2_test(Quad8F Quad8FMain.cxx Quad8F.h)

add_minuit2_test(Quad12F Quad12FMain.cxx Quad12F.h)

# Pass the parent's configuration through to the nested build; otherwise
# ctest picks a platform-dependent default that can mismatch the parent
# (e.g. Debug vs Release with Visual Studio generators)
get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_isMultiConfig)
    set(_exampleBuildConfig --build-config $<CONFIG>)
    set(_exampleBuildType "")
    set(_exampleTestConfig -C $<CONFIG>)
else()
    set(_exampleBuildConfig "")
    set(_exampleBuildType "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
    set(_exampleTestConfig "")
endif()
if(CMAKE_GENERATOR_PLATFORM)
    set(_exampleGeneratorPlatform --build-generator-platform "${CMAKE_GENERATOR_PLATFORM}")
else()
    set(_exampleGeneratorPlatform "")
endif()

add_test(
    NAME ExampleCMakeBuild
    COMMAND "${CMAKE_CTEST_COMMAND}"
            --build-and-test
            "${Minuit2_SOURCE_DIR}/examples/simple/"
            "${CMAKE_CURRENT_BINARY_DIR}/simple/"
            --build-generator "${CMAKE_GENERATOR}"
            ${_exampleGeneratorPlatform}
            ${_exampleBuildConfig}
            --build-options "-DMinuit2_DIR=${Minuit2_BINARY_DIR}" ${_exampleBuildType}
            --test-command "${CMAKE_CTEST_COMMAND}" ${_exampleTestConfig}
    )
