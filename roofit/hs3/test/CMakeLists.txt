ROOT_ADD_GTEST(testRooFitHS3 testRooFitHS3.cxx LIBRARIES RooFitCore RooFit RooFitHS3)
ROOT_ADD_GTEST(testHS3SimultaneousFit testHS3SimultaneousFit.cxx LIBRARIES RooFitCore RooFit RooFitHS3 RooStats)

if(test_roofit_hs3testsuite)
    include(FetchContent)
    FetchContent_Declare(
        hs3testsuite
        GIT_REPOSITORY https://github.com/Phmonski/HS3TestSuite.git
        GIT_TAG 9d04e321ae6fddd283a35507f14ecf852eb7df61
    )
    FetchContent_MakeAvailable(hs3testsuite)
    if(EXISTS ${hs3testsuite_SOURCE_DIR})
        ROOT_ADD_PYUNITTEST(hs3-suite test_hs3_suite.py 
            PYTHON_DEPS jsonschema 
            ENVIRONMENT HS3TESTSUITE_ROOT=${hs3testsuite_SOURCE_DIR}
            COPY_TO_BUILDDIR hs3testsuite_roofit_backend.py
            )
    endif()
endif()