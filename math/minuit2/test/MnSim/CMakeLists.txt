# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.


add_library(GaussSim
    GaussDataGen.cxx
    GaussFcn.cxx
    GaussFcn2.cxx
    )
target_include_directories(GaussSim PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(GaussSim PUBLIC Minuit2)


add_minuit2_test(demoMinimizer demoMinimizer.cxx)

add_minuit2_test(DemoFumili DemoFumili.cxx)
target_link_libraries(DemoFumili PUBLIC GaussSim)

add_minuit2_test(DemoGaussSim DemoGaussSim.cxx)
target_link_libraries(DemoGaussSim PUBLIC GaussSim)

add_minuit2_test(ParallelTest ParallelTest.cxx)

add_minuit2_test(PaulTest PaulTest.cxx)
target_link_libraries(PaulTest PUBLIC GaussSim)

add_minuit2_test(PaulTest2 PaulTest2.cxx)
target_link_libraries(PaulTest2 PUBLIC GaussSim)

add_minuit2_test(PaulTest3 PaulTest3.cxx)
target_link_libraries(PaulTest3 PUBLIC GaussSim)

add_minuit2_test(PaulTest4 PaulTest4.cxx)

add_minuit2_test(ReneTest ReneTest.cxx)

configure_file(paul.txt paul.txt COPYONLY)
configure_file(paul2.txt paul2.txt COPYONLY)
configure_file(paul3.txt paul3.txt COPYONLY)
configure_file(paul4.txt paul4.txt COPYONLY)
