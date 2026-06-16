ROOT_ADD_GTEST(testRooFitHS3 testRooFitHS3.cxx LIBRARIES RooFitCore RooFit RooFitHS3)
ROOT_ADD_GTEST(testHS3SimultaneousFit testHS3SimultaneousFit.cxx LIBRARIES RooFitCore RooFit RooFitHS3 RooStats)

if(test_roofit_hs3testsuite)
    set(hs3testsuite_dir ${CMAKE_CURRENT_SOURCE_DIR}/HS3TestSuite)
    if(NOT EXISTS ${hs3testsuite_dir})
        find_package(Git QUIET REQUIRED)
        execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/Phmonski/HS3TestSuite
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(EXISTS ${hs3testsuite_dir})
        ROOT_ADD_PYUNITTEST(hs3-suite test_hs3_suite.py PYTHON_DEPS jsonschema ENVIRONMENT HS3TESTSUITE_ROOT=${hs3testsuite_dir})
    endif()
endif()