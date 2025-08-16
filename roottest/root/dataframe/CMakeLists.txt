if(NOT dataframe)
  return()
endif()

ROOTTEST_ADD_TESTDIRS()

set(DFLIBRARIES Core RIO Hist Tree MathCore TreePlayer ROOTDataFrame ROOTVecOps ROOTNTuple)

# ROOT-9975
ROOTTEST_ADD_TEST(test_ROOT9975
                  COPY_TO_BUILDDIR ROOT9975.root
                  MACRO test_ROOT9975.C)

# ROOT-9674
ROOTTEST_ADD_TEST(test_readShip
                  COPY_TO_BUILDDIR ship_ROOT_9674.root
                  MACRO readShip.C)

# Linked to ROOT-9773: train cache with TTreeReader
ROOTTEST_ADD_TEST(test_trainCache
                  COPY_TO_BUILDDIR bigFile.root
                  MACRO test_trainCache.C
                  OUTREF test_trainCache.ref)

if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64) OR M1_BROKEN_TESTS)
  # ROOT-9628
  ROOTTEST_ADD_TEST(test_chainZombieFile
                    COPY_TO_BUILDDIR input1.root input2.root
                    ${WILLFAIL_ON_WIN32}
                    MACRO test_chainZombieFile.C)
endif()

# ROOT-9366
# We use this to write something like the FCC data model, dictionaries done with aclic
ROOTTEST_ADD_TEST(writeFcc
                  MACRO writeFcc.C+
                  FIXTURES_SETUP root-dataframe-writeFcc-fixture)

# We build and run an executable that reads the file created above without dictionaries
ROOTTEST_GENERATE_EXECUTABLE(test_readFcc
                             readFcc.C
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-readFcc-fixture)

ROOTTEST_ADD_TEST(test_readFcc
                  EXEC ./test_readFcc
                  COPY_TO_BUILDDIR fccSkim.root
                  FIXTURES_REQUIRED root-dataframe-writeFcc-fixture
                                    root-dataframe-readFcc-fixture)

# ROOT-9116
ROOTTEST_GENERATE_EXECUTABLE(test_gdirectoryRestore
                             test_gdirectoryRestore.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-gdirectoryRestore-fixture)

ROOTTEST_ADD_TEST(test_gdirectoryRestore
                  EXEC ./test_gdirectoryRestore
                  FIXTURES_REQUIRED root-dataframe-gdirectoryRestore-fixture)

ROOTTEST_ADD_TEST(templateRecursionLimit
                  MACRO test_templateRecursionLimit.C)

if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64) OR M1_BROKEN_TESTS)
  ROOTTEST_ADD_TEST(missingBranches
                    MACRO test_missingBranches.C
                    ERRREF test_missingBranches.eref
                    ${WILLFAIL_ON_WIN32})
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
ROOTTEST_GENERATE_EXECUTABLE(test_hugeRDF test_hugeRDF.cxx
                             COMPILE_FLAGS "-O2"  # too slow otherwise
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-hugeRDF-fixture)

ROOTTEST_ADD_TEST(test_hugeRDF
                  EXEC ./test_hugeRDF
                  LABELS longtest
                  FIXTURES_REQUIRED root-dataframe-hugeRDF-fixture)
endif()

ROOTTEST_GENERATE_EXECUTABLE(test_snapshot_manytasks
                             test_snapshot_manytasks.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-snapshot_manytasks-fixture)

ROOTTEST_ADD_TEST(test_snapshot_manytasks
                  EXEC ./test_snapshot_manytasks
                  FIXTURES_REQUIRED root-dataframe-snapshot_manytasks-fixture)

ROOTTEST_GENERATE_EXECUTABLE(test_columnoverride
                             test_columnoverride.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-columnoverride-fixture)

ROOTTEST_ADD_TEST(test_columnoverride
                  EXEC ./test_columnoverride
                  FIXTURES_REQUIRED root-dataframe-columnoverride-fixture)

ROOTTEST_GENERATE_EXECUTABLE(regression_emptysource
                             regression_emptysource.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-regression_emptysource-fixture)

ROOTTEST_ADD_TEST(regression_emptysource
                  EXEC ./regression_emptysource
                  FIXTURES_REQUIRED root-dataframe-regression_emptysource-fixture)

ROOTTEST_ADD_TEST(test_snapshotNFiles
                  MACRO test_snapshotNFiles.C)

ROOTTEST_ADD_TEST(test_listFilesCtor
                  MACRO test_listFilesCtor.C)

ROOTTEST_GENERATE_EXECUTABLE(emptysource
                             test_emptysource.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-emptysource-fixture)

ROOTTEST_ADD_TEST(emptysource
                  EXEC ./emptysource
                  OUTREF test_emptysource.ref
                  FIXTURES_REQUIRED root-dataframe-emptysource-fixture)

ROOTTEST_ADD_TEST(regression_snapshot
                  MACRO regression_snapshot.C+
                  OUTREF regression_snapshot.ref)

ROOTTEST_GENERATE_EXECUTABLE(test_inference
                             test_inference.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-test_inference-fixture)

ROOTTEST_ADD_TEST(test_inference
                  EXEC ./test_inference
                  OUTREF test_inference.ref
                  FIXTURES_REQUIRED root-dataframe-test_inference-fixture)

ROOTTEST_ADD_TEST(test_snapshot_dir
                  MACRO test_snapshot_dir.C+
                  ${WILLFAIL_ON_WIN32}
                  OUTREF test_snapshot_dir.ref)

ROOTTEST_ADD_TEST(test_snapshot_subdir
                  MACRO test_snapshot_subdir.C+
                  ${WILLFAIL_ON_WIN32}
                  OUTREF test_snapshot_subdir.ref)

ROOTTEST_ADD_TEST(test_snapshotMT_dir
                  MACRO test_snapshotMT_dir.C+
                  ${WILLFAIL_ON_WIN32}
                  OUTREF test_snapshotMT_dir.ref)

ROOTTEST_ADD_TEST(test_snapshotMT_subdir
                  MACRO test_snapshotMT_subdir.C+
                  ${WILLFAIL_ON_WIN32}
                  OUTREF test_snapshotMT_subdir.ref)

ROOTTEST_ADD_TEST(test_stringfiltercolumn
                  MACRO test_stringfiltercolumn.C+
                  OUTREF test_stringfiltercolumn.ref)

ROOTTEST_ADD_TEST(test_glob
                  MACRO test_glob.C+
                  OUTREF test_glob.ref)

ROOTTEST_ADD_TEST(test_reduce
                  MACRO test_reduce.C+
                  OUTREF test_reduce.ref)

ROOTTEST_GENERATE_EXECUTABLE(test_callables
                             test_callables.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-test_callables-fixture)

ROOTTEST_ADD_TEST(test_callables
                  EXEC ./test_callables
                  OUTREF test_callables.ref
                  FIXTURES_REQUIRED root-dataframe-test_callables-fixture)

ROOTTEST_GENERATE_EXECUTABLE(testIMT
                             testIMT.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-testIMT-fixture)

ROOTTEST_ADD_TEST(testIMT
                  EXEC ./testIMT
                  OUTREF testIMT.ref
                  FIXTURES_REQUIRED root-dataframe-testIMT-fixture)

if(MSVC AND MSVC_VERSION LESS 1924 AND NOT win_broken_tests)
  # FIXME: with Visual Studio v16.3, test_readerarray fails in debug mode, so compile it in release mode
  # this is fixed with more recent versions of Visual Studio
  set(_test_flags ${CMAKE_CXX_FLAGS_RELEASE})
endif()
ROOTTEST_GENERATE_EXECUTABLE(test_readerarray
                             test_readerarray.cxx
                             COMPILE_FLAGS ${_test_flags}
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-test_readerarray-fixture)

ROOTTEST_ADD_TEST(test_readerarray
                  EXEC ./test_readerarray
                  OUTREF test_readerarray.ref
                  FIXTURES_REQUIRED root-dataframe-test_readerarray-fixture)

ROOTTEST_GENERATE_EXECUTABLE(typeguessing
                             test_typeguessing.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-typeguessing-fixture)

ROOTTEST_ADD_TEST(typeguessing
                  EXEC ./typeguessing
                  OUTREF test_typeguessing.ref
                  FIXTURES_REQUIRED root-dataframe-typeguessing-fixture)

ROOTTEST_GENERATE_EXECUTABLE(misc
                             test_misc.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-misc-fixture)

ROOTTEST_ADD_TEST(misc
                  EXEC ./misc
                  OUTREF test_misc.ref
                  FIXTURES_REQUIRED root-dataframe-misc-fixture)

ROOTTEST_ADD_TEST(ctors
                  MACRO test_ctors.C+
                  OUTREF test_ctors.ref)

ROOTTEST_GENERATE_EXECUTABLE(branchoverwrite
                             test_branchoverwrite.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-branchoverwrite-fixture)

ROOTTEST_ADD_TEST(branchoverwrite
                  EXEC ./branchoverwrite
                  OUTREF test_branchoverwrite.ref
                  FIXTURES_REQUIRED root-dataframe-branchoverwrite-fixture)

ROOTTEST_GENERATE_EXECUTABLE(regression_multipletriggerrun
                             regression_multipletriggerrun.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-regression_multipletriggerrun-fixture)

ROOTTEST_ADD_TEST(regression_multipletriggerrun
                  EXEC ./regression_multipletriggerrun
                  OUTREF regression_multipletriggerrun.ref
                  FIXTURES_REQUIRED root-dataframe-regression_multipletriggerrun-fixture)

ROOTTEST_GENERATE_EXECUTABLE(regression_zeroentries
                             regression_zeroentries.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-regression_zeroentries-fixture)

ROOTTEST_ADD_TEST(regression_zeroentries
                  EXEC ./regression_zeroentries
                  FIXTURES_REQUIRED root-dataframe-regression_zeroentries-fixture)

ROOTTEST_GENERATE_EXECUTABLE(foreach
                             test_foreach.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-foreach-fixture)
ROOTTEST_ADD_TEST(foreach
                  EXEC ./foreach
                  OUTREF test_foreach.ref
                  FIXTURES_REQUIRED root-dataframe-foreach-fixture)

ROOTTEST_GENERATE_EXECUTABLE(reports
                             test_reports.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-reports-fixture)

ROOTTEST_ADD_TEST(reports
                  EXEC ./reports
                  OUTREF test_reports.ref
                  FIXTURES_REQUIRED root-dataframe-reports-fixture)

ROOTTEST_GENERATE_EXECUTABLE(par
                             test_par.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-par-fixture)

ROOTTEST_ADD_TEST(par
                  EXEC ./par
                  OUTREF test_par.ref
                  FIXTURES_REQUIRED root-dataframe-par-fixture)

ROOTTEST_GENERATE_EXECUTABLE(read_leaves
                             test_read_leaves.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-read_leaves-fixture)

ROOTTEST_ADD_TEST(read_leaves
                  EXEC ./read_leaves
                  COPY_TO_BUILDDIR test_read_leaves.h
                  FIXTURES_REQUIRED root-dataframe-read_leaves-fixture)

ROOTTEST_GENERATE_EXECUTABLE(read_leaves_nodict
                             test_read_leaves_nodict.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-read_leaves_nodict-fixture)

ROOTTEST_ADD_TEST(read_leaves_nodict
                  EXEC ./read_leaves_nodict
                  FIXTURES_REQUIRED root-dataframe-read_leaves_nodict-fixture)

ROOTTEST_ADD_TEST(test_readTotemNtuple
                  MACRO test_readTotemNtuple.C
                  COPY_TO_BUILDDIR Slimmed_TotemNTuple_9883.040.ntuple.root
                  OUTREF test_readTotemNtuple.ref)

ROOTTEST_GENERATE_EXECUTABLE(test_progressiveCSV
                             test_progressiveCSV.cxx
                             LIBRARIES ${DFLIBRARIES}
                             FIXTURES_SETUP root-dataframe-test_progressiveCSV-fixture)

ROOTTEST_ADD_TEST(test_progressiveCSV
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_progressiveCSV.sh
                  COPY_TO_BUILDDIR test_progressiveCSV.csv
                  OUTREF test_progressiveCSV.ref
                  FIXTURES_REQUIRED root-dataframe-test_progressiveCSV-fixture)

# need the '+' to autogenerate required dictionaries with ACLiC
ROOTTEST_ADD_TEST(test_nested_rvec_snapshot
                  MACRO test_nested_rvec_snapshot.C+)

ROOTTEST_ADD_TEST(test_snapshot_copyaddresses
                  MACRO test_snapshot_copyaddresses.C+)

ROOT_ADD_GTEST(test_norootextension test_norootextension.cxx LIBRARIES ROOT::ROOTDataFrame)
