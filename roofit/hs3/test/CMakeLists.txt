ROOT_ADD_GTEST(testRooFitHS3 testRooFitHS3.cxx LIBRARIES RooFitCore RooFit RooFitHS3 RooStats)

ROOT_ADD_GTEST(testJSONInterface testJSONInterface.cxx LIBRARIES RooFitHS3 Matrix)

if(test_roofit_hs3testsuite)
    # The HS3 conformance test suite is split into two repositories: the frozen fixtures
    # (HS3TestSuite) and the Python runner (HS3TestSuiteCode). Both are pinned to explicit
    # commits. See roofit/doc/developers/roofit_hs3.md ("Conformance testing") for how to
    # bump them together with hs3testsuite_expected_failures.txt.
    set(hs3testsuite_url https://github.com/hep-statistics-serialization-standard/HS3TestSuite.git)
    set(hs3testsuite_tag 9f799b27f63b40d1ba96ff66d14d61787892606d)
    set(hs3testsuitecode_url https://github.com/hep-statistics-serialization-standard/HS3TestSuiteCode.git)
    set(hs3testsuitecode_tag 36aeb1c67b14dbbd9d9e9b79a4bad561c5e6dac7)

    include(FetchContent)
    # suppress git interactive prompt so fetch fails fast if the repo is gone or unreachable
    set(ENV{GIT_TERMINAL_PROMPT} "0")
    foreach(repo hs3testsuite hs3testsuitecode)
        string(TOUPPER ${repo} repo_upper)
        # Probe the network first if the sources still have to be downloaded, so that an
        # offline configuration fails with an actionable message instead of a git error
        # from the FetchContent sub-build.
        if(NOT FETCHCONTENT_FULLY_DISCONNECTED AND NOT FETCHCONTENT_SOURCE_DIR_${repo_upper}
           AND NOT EXISTS ${FETCHCONTENT_BASE_DIR}/${repo}-src/.git)
            execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --exit-code ${${repo}_url} HEAD
                            RESULT_VARIABLE ls_remote_result OUTPUT_QUIET ERROR_QUIET TIMEOUT 60)
            if(NOT ls_remote_result EQUAL 0)
                message(FATAL_ERROR "-Dtest_roofit_hs3testsuite=ON requires network access at configuration "
                                    "time, but ${${repo}_url} could not be reached. Configure with network "
                                    "access, provide a local checkout of commit ${${repo}_tag} with "
                                    "-DFETCHCONTENT_SOURCE_DIR_${repo_upper}=<path>, or disable the HS3 test "
                                    "suite with -Dtest_roofit_hs3testsuite=OFF.")
            endif()
        endif()
        FetchContent_Declare(${repo} GIT_REPOSITORY ${${repo}_url} GIT_TAG ${${repo}_tag})
    endforeach()
    FetchContent_MakeAvailable(hs3testsuite hs3testsuitecode)

    ROOT_ADD_PYUNITTEST(hs3-suite test_hs3_suite.py
        PYTHON_DEPS jsonschema
        ENVIRONMENT HS3TESTSUITE_ROOT=${hs3testsuite_SOURCE_DIR}
                    HS3TESTSUITECODE_ROOT=${hs3testsuitecode_SOURCE_DIR}
                    HS3TESTSUITE_EXPECTED_FAILURES=${CMAKE_CURRENT_SOURCE_DIR}/hs3testsuite_expected_failures.txt
        COPY_TO_BUILDDIR hs3testsuite_roofit_backend.py
        )
endif()
