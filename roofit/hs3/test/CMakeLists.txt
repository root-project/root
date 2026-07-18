ROOT_ADD_GTEST(testRooFitHS3 testRooFitHS3.cxx LIBRARIES RooFitCore RooFit RooFitHS3 RooStats)

ROOT_ADD_GTEST(testJSONInterface testJSONInterface.cxx LIBRARIES RooFitHS3 Matrix)

if(test_roofit_hs3testsuite)
    include(FetchContent)
    FetchContent_Declare(
        hs3testsuite
        GIT_REPOSITORY https://github.com/hep-statistics-serialization-standard/HS3TestSuite.git
        GIT_TAG b2e5f19023025d03b7d2239154a02e69f43a1374
    )
    # suppress git interactive prompt so fetch fails fast if the repo is gone or unreachable
    set(ENV{GIT_TERMINAL_PROMPT} "0")
    FetchContent_MakeAvailable(hs3testsuite)
    ROOT_ADD_PYUNITTEST(hs3-suite test_hs3_suite.py 
        PYTHON_DEPS jsonschema 
        ENVIRONMENT HS3TESTSUITE_ROOT=${hs3testsuite_SOURCE_DIR}
        COPY_TO_BUILDDIR hs3testsuite_roofit_backend.py
        )
endif()
