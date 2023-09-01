# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.


add_minuit2_test(Quad1F Quad1FMain.cxx Quad1F.h)

add_minuit2_test(Quad4F Quad4FMain.cxx Quad4F.h)

add_minuit2_test(Quad8F Quad8FMain.cxx Quad8F.h)

add_minuit2_test(Quad12F Quad12FMain.cxx Quad12F.h)

add_test(
    NAME ExampleCMakeBuild
    COMMAND "${CMAKE_CTEST_COMMAND}"
            --build-and-test
            "${Minuit2_SOURCE_DIR}/examples/simple/"
            "${CMAKE_CURRENT_BINARY_DIR}/simple/"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "${CMAKE_CTEST_COMMAND}"
    )
