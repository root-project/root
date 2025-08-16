if(UNIX AND 32BIT)
  # linux32 llvm-JIT ABI issue constructing the pair of map<tthread::id, uint>.
  # Compile instead.
  ROOTTEST_ADD_TEST(TThreadedObjectAutoMemManagement
                    MACRO  testTThreadedObjectAutoMemManagement.C+)
else()
  ROOTTEST_ADD_TEST(TThreadedObjectAutoMemManagement
                    MACRO  testTThreadedObjectAutoMemManagement.C)
endif()

if(imt)

  # ROOT-8850
  ROOTTEST_GENERATE_EXECUTABLE(loadImt
                               loadImt.cpp
                               LIBRARIES Core RIO
                               FIXTURES_SETUP root-multicore-loadImt-fixture)

  ROOTTEST_ADD_TEST(loadImt
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/loadImt
                    FIXTURES_REQUIRED root-multicore-loadImt-fixture)

endif()

ROOTTEST_GENERATE_EXECUTABLE(testSetAddress
                             testSetAddress.cpp
                             LIBRARIES Core MathCore RIO Tree
                             FIXTURES_SETUP root-multicore-testSetAddress-fixture)

if(MSVC)
  ROOTTEST_ADD_TEST(testSetAddressLoop
                    EXEC ${CMAKE_CURRENT_SOURCE_DIR}/testSetAddressLoop.bat
                    COPY_TO_BUILDDIR just1entry.root
                    FIXTURES_REQUIRED root-multicore-testSetAddress-fixture)
else()
  ROOTTEST_ADD_TEST(testSetAddressLoop
                    EXEC ${CMAKE_CURRENT_SOURCE_DIR}/testSetAddressLoop.sh
                    COPY_TO_BUILDDIR just1entry.root
                    FIXTURES_REQUIRED root-multicore-testSetAddress-fixture)
endif()

ROOTTEST_ADD_TEST(checkTThreadedObjectCopies
                  MACRO checkTThreadedObjectCopies.C)

ROOTTEST_GENERATE_EXECUTABLE(tSequentialExecutor
                             tSequentialExecutor.cpp
                             LIBRARIES Core MathCore Hist
                             FIXTURES_SETUP root-multicore-tSequentialExecutor-fixture)

ROOTTEST_ADD_TEST(SequentialExecutor
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tSequentialExecutor
                  FIXTURES_REQUIRED root-multicore-tSequentialExecutor-fixture)

if(NOT MSVC AND NOT dev)
  ROOTTEST_GENERATE_EXECUTABLE(processExecutorH1Test
                               tProcessExecutorH1Test.cpp
                               COMPILE_FLAGS "-I${ROOTSYS}/tutorials/legacy/multicore"
                               LIBRARIES ROOT::MultiProc ROOT::Core ROOT::Net ROOT::TreePlayer ROOT::Tree ROOT::RIO ROOT::Hist ROOT::Gpad ROOT::Graf
                               FIXTURES_SETUP root-multicore-processExecutorH1Test-fixture)

  ROOTTEST_ADD_TEST(processExecutorH1Test
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorH1Test
                    OUTREF tProcessExecutorH1Test.ref
                    FIXTURES_REQUIRED root-multicore-processExecutorH1Test-fixture)

  ROOTTEST_GENERATE_EXECUTABLE(processExecutorTreeTest
                               tProcessExecutorTreeTest.cpp
                               LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist
                               FIXTURES_SETUP root-multicore-processExecutorTreeTest-fixture)

  ROOTTEST_ADD_TEST(processExecutorTreeTest
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorTreeTest
                    OUTREF tProcessExecutorTreeTest.ref
                    FIXTURES_REQUIRED root-multicore-processExecutorTreeTest-fixture)

  ROOTTEST_GENERATE_EXECUTABLE(processExecutorProcessTest
                               tProcessExecutorProcessTest.cpp
                               LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist
                               FIXTURES_SETUP root-multicore-processExecutorProcessTest-fixture)

  ROOTTEST_ADD_TEST(processExecutorProcess
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorProcessTest
                    OUTREF tProcessExecutorProcessTest.ref
                    FIXTURES_REQUIRED root-multicore-processExecutorProcessTest-fixture)

  ROOTTEST_GENERATE_EXECUTABLE(processExecutor
                               tProcessExecutor.cpp
                               LIBRARIES MultiProc Core Net TreePlayer RIO Hist MathCore
                               FIXTURES_SETUP root-multicore-processExecutor-fixture)

  ROOTTEST_ADD_TEST(processExecutor
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutor
                    FIXTURES_REQUIRED root-multicore-processExecutor-fixture)
endif()

if(imt)
  include_directories(${TBB_INCLUDE_DIRS})
  ROOTTEST_GENERATE_EXECUTABLE(threadExecutor
                               tThreadExecutor.cpp
                               LIBRARIES Core Imt Hist MathCore
                               FIXTURES_SETUP root-multicore-threadExecutor-fixture)

  ROOTTEST_ADD_TEST(threadExecutor
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/threadExecutor
                    FIXTURES_REQUIRED root-multicore-threadExecutor-fixture)
endif()

if(NOT MSVC)
   ROOTTEST_GENERATE_EXECUTABLE(tExecutorMP
                                tExecutorMP.cxx
                                LIBRARIES Core MultiProc Imt Net TreePlayer RIO Hist MathCore
                                FIXTURES_SETUP root-multicore-tExecutorMP-fixture)

   ROOTTEST_ADD_TEST(tExecutorMP
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/tExecutorMP
                     FIXTURES_REQUIRED root-multicore-tExecutorMP-fixture)

   set(MULTIPROCLIB MultiProc)
else()
   # TProcessExecutor and the MultiProc library are not available on windows
   set(MULTIPROCLIB "")
endif()

ROOTTEST_GENERATE_EXECUTABLE(tExecutorMT
                             tExecutorMT.cxx
                             LIBRARIES Core ${MULTIPROCLIB} Imt Net TreePlayer RIO Hist MathCore
                             FIXTURES_SETUP root-multicore-tExecutorMT-fixture)

ROOTTEST_ADD_TEST(tExecutorMT
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tExecutorMT
                  FIXTURES_REQUIRED root-multicore-tExecutorMT-fixture)

ROOTTEST_GENERATE_EXECUTABLE(current_dir
                             current_dir.cpp
                             LIBRARIES Core RIO Thread
                             FIXTURES_SETUP root-multicore-current_dir-fixture)

ROOTTEST_ADD_TEST(current_dir
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/current_dir
                  OUTREF current_dir.ref
                  FIXTURES_REQUIRED root-multicore-current_dir-fixture)

ROOTTEST_ADD_TEST(exectstformula
                  MACRO exectstformula.C
                  OUTREF exectstformula.ref)

ROOTTEST_ADD_TEST(exectsautoparse
                  MACRO exectsautoparse.C
                  OUTREF exectsautoparse.ref)

ROOTTEST_ADD_TEST(exectsinclude
                 MACRO  exectsinclude.C
                 OUTCNVCMD grep -v -e "Line processed"
                 OUTREF exectsinclude.ref)

ROOTTEST_ADD_TEST(execautoloadts
                 MACRO  execautoloadts.C
                 OUTREF execautoloadts.ref)

ROOTTEST_GENERATE_EXECUTABLE(tclass_methods_test
                             tclass_methods_test.cpp
                             LIBRARIES Core Thread
                             FIXTURES_SETUP root-multicore-tclass_methods_test-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tclass_methods
                                    tclass_methods.h
                                    SELECTION tclass_methods_selection.xml
                                    FIXTURES_SETUP root-multicore-tclass_methods-fixture)

ROOTTEST_ADD_TEST(tclass_methods
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tclass_methods_test
                  FIXTURES_REQUIRED root-multicore-tclass_methods_test-fixture root-multicore-tclass_methods-fixture
                  FIXTURES_SETUP root-multicore-tclass_methodstest-fixture)

ROOTTEST_GENERATE_EXECUTABLE(ttree_write
                             ttree_write.cpp
                             LIBRARIES Core Hist RIO Thread Tree
                             FIXTURES_SETUP root-multicore-ttree_write-fixture)

ROOTTEST_ADD_TEST(ttree_write
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_write
                  FIXTURES_REQUIRED root-multicore-ttree_write-fixture
                  LABELS longtest)

ROOTTEST_GENERATE_EXECUTABLE(ttree_read
                             ttree_read.cpp
                             LIBRARIES Core Hist RIO Thread Tree
                             FIXTURES_SETUP root-multicore-ttree_read-fixture)

ROOTTEST_ADD_TEST(ttree_read
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_read
                  COPY_TO_BUILDDIR ttree_read.root
                  FIXTURES_REQUIRED root-multicore-ttree_read-fixture)

ROOTTEST_GENERATE_EXECUTABLE(th1f_fill
                             th1f_fill.cpp
                             LIBRARIES Core Hist RIO Thread Tree
                             FIXTURES_SETUP root-multicore-th1f_fill-fixture)

ROOTTEST_ADD_TEST(th1f_fill
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/th1f_fill
                  COPY_TO_BUILDDIR file1.root file2.root file3.root
                  OUTREF execth1f_fill.ref
                  FIXTURES_REQUIRED root-multicore-th1f_fill-fixture)

# ROOTTEST_GENERATE_EXECUTABLE(tformula
#                              tformula.cpp
#                              LIBRARIES Core Hist Thread
#                              FIXTURES_SETUP root-multicore-tformula-fixture)

# ROOTTEST_ADD_TEST(tformula
#                   EXEC ${CMAKE_CURRENT_BINARY_DIR}/tformula
#                   FIXTURES_REQUIRED root-multicore-tformula-fixture)

ROOTTEST_GENERATE_EXECUTABLE(tprofile
                             tprofile.cpp
                             LIBRARIES Core Hist Thread
                             FIXTURES_SETUP root-multicore-tprofile-fixture)

ROOTTEST_ADD_TEST(tprofile
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tprofile
                  OUTREF exectprofile.ref
                  FIXTURES_REQUIRED root-multicore-tprofile-fixture)

# Re-Enabled Disable test: see ROOT-7158
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_GENERATE_REFLEX_DICTIONARY(tsenums
                                      tsenums.h
                                      SELECTION tsenums_selection.xml
                                      FIXTURES_SETUP root-multicore-tsenums-fixture)

  ROOTTEST_ADD_TEST(tsenums
                    MACRO  exectsenums.C
                    OUTREF exectsenums.ref
                    FIXTURES_REQUIRED root-multicore-tsenums-fixture
                    FIXTURE_SETUP root-multicore-tsenumstest-fixture)
endif()

ROOTTEST_ADD_TEST(semaphore
                 MACRO  tSemaphore.C
                 OUTCNVCMD grep -v -e  "tSemaphore"
                 OUTREF tSemaphore.ref)

if(OPENGL_gl_LIBRARY AND ROOTTEST_OS_ID MATCHES Scientific|CentOS|Ubuntu|Fedora)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=${OPENGL_gl_LIBRARY})
endif()

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_GENERATE_EXECUTABLE(fork
                               fork.cpp
                               LIBRARIES Core
                               FIXTURES_SETUP root-multicore-fork-fixture)

  ROOTTEST_ADD_TEST(fork
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/fork
                    COPY_TO_BUILDDIR commands1.txt commands2.txt
                    FAILREGEX "Error in" "cannot load any more object with static TLS"
                    FIXTURES_REQUIRED root-multicore-fork-fixture
                                      root-multicore-tsenumstest-fixture
                                      root-multicore-tclass_methodstest-fixture)
endif()

if(imt)
   ROOTTEST_GENERATE_EXECUTABLE(ttree_read_imt
                                ttree_read_imt.cpp
                                LIBRARIES Core Imt Thread Tree RIO
                                FIXTURES_SETUP root-multicore-ttree_read_imt-fixture)

   ROOTTEST_ADD_TEST(ttree_read_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ttree_read_imt.sh
                     FAILREGEX "ERROR"
                     OUTREF ttree_read_imt.ref
                     FIXTURES_REQUIRED root-multicore-ttree_read_imt-fixture)

### Keep this test for debugging purposes: even if ttree_read_imt makes it redundant, it can be useful to
### debug issues that are hard to reproduce locally but can eventually be observed on the test machines.
#   ROOTTEST_GENERATE_EXECUTABLE(ttree_read_imt_allpar
#                                ttree_read_imt_allpar.cpp
#                                LIBRARIES Core Thread Tree
#                                FIXTURES_SETUP root-multicore-ttree_read_imt_allpar-fixture)

#   ROOTTEST_ADD_TEST(ttree_read_imt_allpar
#                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ttree_read_imt_allpar.sh
#                     FAILREGEX "ERROR"
#                     OUTREF ttree_read_imt_allpar.ref
#                     FIXTURES_REQUIRED root-multicore-ttree_read_imt_allpar-fixture)

   ROOTTEST_GENERATE_EXECUTABLE(ttree_write_imt
                                ttree_write_imt.cpp
                                LIBRARIES Core Imt Thread Tree Hist RIO
                                FIXTURES_SETUP root-multicore-ttree_write_imt-fixture)

   ROOTTEST_ADD_TEST(ttree_write_imt
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_write_imt
                     FIXTURES_REQUIRED root-multicore-ttree_write_imt-fixture)

   ROOTTEST_GENERATE_EXECUTABLE(tp_process_imt
                                tp_process_imt.cpp
                                LIBRARIES Core Imt Thread Tree TreePlayer
                                FIXTURES_SETUP root-multicore-tp_process_imt-fixture)

   ROOTTEST_ADD_TEST(tp_process_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tp_process_imt.sh
                     COPY_TO_BUILDDIR tp_process_imt.root
                     FAILREGEX "ERROR"
                     OUTREF tp_process_imt.ref
                     FIXTURES_REQUIRED root-multicore-tp_process_imt-fixture)

   ROOTTEST_GENERATE_EXECUTABLE(tp_process_imt_race_regression
                                tp_process_imt_race_regression.cpp
                                LIBRARIES Core Imt Thread Tree TreePlayer
                                FIXTURES_SETUP root-multicore-tp_process_imt_race_regression-fixture)

   ROOTTEST_ADD_TEST(tp_process_imt_race_regression
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/tp_process_imt_race_regression
                     COPY_TO_BUILDDIR treeprocmt_race_regression_input1.root treeprocmt_race_regression_input2.root
                                      treeprocmt_race_regression_input3.root treeprocmt_race_regression_input4.root
                     FIXTURES_REQUIRED root-multicore-tp_process_imt_race_regression-fixture)
endif()
