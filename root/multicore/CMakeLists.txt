ROOTTEST_ADD_TESTDIRS()

# Workaround (k)ubuntu bug: runtime issue when using pthread and gcc48
if(ROOTTEST_OS_ID MATCHES Ubuntu)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(UNIX AND 32BIT)
  # linux32 llvm-JIT ABI issue constructing the pair of map<tthread::id, uint>.
  # Compile instead.
  ROOTTEST_ADD_TEST(TThreadedObjectAutoMemManagement
                    MACRO  testTThreadedObjectAutoMemManagement.C+)
else()
  ROOTTEST_ADD_TEST(TThreadedObjectAutoMemManagement
                    MACRO  testTThreadedObjectAutoMemManagement.C)
endif()

if(ROOT_imt_FOUND)

# ROOT-8850
ROOTTEST_GENERATE_EXECUTABLE(loadImt loadImt.cpp LIBRARIES Core RIO)

ROOTTEST_ADD_TEST(loadImt
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/loadImt
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

endif()

ROOTTEST_GENERATE_EXECUTABLE(testSetAddress testSetAddress.cpp LIBRARIES Core MathCore RIO Tree)

if(MSVC)
  ROOTTEST_ADD_TEST(testSetAddressLoop
                    EXEC ${CMAKE_CURRENT_SOURCE_DIR}/testSetAddressLoop.bat
                    COPY_TO_BUILDDIR just1entry.root
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})
else()
  ROOTTEST_ADD_TEST(testSetAddressLoop
                    EXEC ${CMAKE_CURRENT_SOURCE_DIR}/testSetAddressLoop.sh
                    COPY_TO_BUILDDIR just1entry.root
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()

ROOTTEST_ADD_TEST(checkTThreadedObjectCopies
                  MACRO checkTThreadedObjectCopies.C)

ROOTTEST_GENERATE_EXECUTABLE(tSequentialExecutor tSequentialExecutor.cpp LIBRARIES Core MathCore Hist)

ROOTTEST_ADD_TEST(SequentialExecutor
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tSequentialExecutor
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# Workaround for gcc 5.x (x<4) bug in optimized mode
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
  set(PROCESSOREXECUTORH1TEST_EXE_COMPILE_FLAGS "-O1 -I${ROOTSYS}/tutorials/multicore")
else()
  set(PROCESSOREXECUTORH1TEST_EXE_COMPILE_FLAGS "-I${ROOTSYS}/tutorials/multicore")
endif()

if(NOT MSVC OR win_broken_tests)

  if(MSVC AND DEFINED ROOT_SOURCE_DIR)
    ROOTTEST_GENERATE_EXECUTABLE(processExecutorH1Test tProcessExecutorH1Test.cpp COMPILE_FLAGS ${PROCESSOREXECUTORH1TEST_EXE_COMPILE_FLAGS} LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist Gpad Graf)
  else()
    ROOTTEST_GENERATE_EXECUTABLE(processExecutorH1Test tProcessExecutorH1Test.cpp COMPILE_FLAGS ${PROCESSOREXECUTORH1TEST_EXE_COMPILE_FLAGS} LIBRARIES ROOT::MultiProc ROOT::Core ROOT::Net ROOT::TreePlayer ROOT::Tree ROOT::RIO ROOT::Hist ROOT::Gpad ROOT::Graf)
  endif()

  ROOTTEST_ADD_TEST(processExecutorH1Test
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorH1Test
                    OUTREF tProcessExecutorH1Test.ref
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})

  ROOTTEST_GENERATE_EXECUTABLE(processExecutorTreeTest tProcessExecutorTreeTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

  ROOTTEST_ADD_TEST(processExecutorTreeTest
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorTreeTest
                    OUTREF tProcessExecutorTreeTest.ref
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})

  ROOTTEST_GENERATE_EXECUTABLE(processExecutorProcessTest tProcessExecutorProcessTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

  ROOTTEST_ADD_TEST(processExecutorProcess
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutorProcessTest
                    OUTREF tProcessExecutorProcessTest.ref
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})

  ROOTTEST_GENERATE_EXECUTABLE(processExecutor tProcessExecutor.cpp LIBRARIES MultiProc Core Net TreePlayer RIO Hist MathCore)

  ROOTTEST_ADD_TEST(processExecutor
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/processExecutor
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()

if(ROOT_imt_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  ROOTTEST_GENERATE_EXECUTABLE(threadExecutor tThreadExecutor.cpp LIBRARIES Core Imt Hist MathCore)

  ROOTTEST_ADD_TEST(threadExecutor
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/threadExecutor
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()

if(NOT MSVC)
   ROOTTEST_GENERATE_EXECUTABLE(tExecutorMP tExecutorMP.cxx LIBRARIES Core MultiProc Imt Net TreePlayer RIO Hist MathCore)
   ROOTTEST_ADD_TEST(tExecutorMP
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/tExecutorMP
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

   set(MULTIPROCLIB MultiProc)
else()
   # TProcessExecutor and the MultiProc library are not available on windows
   set(MULTIPROCLIB "")
endif()

ROOTTEST_GENERATE_EXECUTABLE(tExecutorMT tExecutorMT.cxx LIBRARIES Core ${MULTIPROCLIB} Imt Net TreePlayer RIO Hist MathCore)
ROOTTEST_ADD_TEST(tExecutorMT
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tExecutorMT
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(current_dir current_dir.cpp LIBRARIES Core RIO Thread)

ROOTTEST_ADD_TEST(current_dir
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/current_dir
                  OUTREF current_dir.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

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

ROOTTEST_GENERATE_EXECUTABLE(tclass_methods_test tclass_methods_test.cpp LIBRARIES Core Thread)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tclass_methods tclass_methods.h  SELECTION tclass_methods_selection.xml)

ROOTTEST_ADD_TEST(tclass_methods
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tclass_methods_test
                  DEPENDS ${GENERATE_EXECUTABLE_TEST} ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_EXECUTABLE(ttree_write ttree_write.cpp LIBRARIES Core Hist RIO Thread Tree)

ROOTTEST_ADD_TEST(ttree_write
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_write
                  DEPENDS ${GENERATE_EXECUTABLE_TEST}
                  LABELS longtest)

ROOTTEST_GENERATE_EXECUTABLE(ttree_read ttree_read.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(ttree_read.root . COPYONLY)

ROOTTEST_ADD_TEST(ttree_read
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_read
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(th1f_fill th1f_fill.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(file1.root . COPYONLY)
configure_file(file2.root . COPYONLY)
configure_file(file3.root . COPYONLY)

ROOTTEST_ADD_TEST(th1f_fill
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/th1f_fill
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# ROOTTEST_GENERATE_EXECUTABLE(tformula tformula.cpp LIBRARIES Core Hist Thread)
#
# ROOTTEST_ADD_TEST(tformula
#                   EXEC ./tformula
#                   DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(tprofile tprofile.cpp LIBRARIES Core Hist Thread)

ROOTTEST_ADD_TEST(tprofile
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/tprofile
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# Re-Enabled Disable test: see ROOT-7158
ROOTTEST_GENERATE_REFLEX_DICTIONARY(tsenums tsenums.h  SELECTION tsenums_selection.xml)

ROOTTEST_ADD_TEST(tsenums
                  MACRO  exectsenums.C
                  OUTREF exectsenums.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_GENERATE_EXECUTABLE(fork fork.cpp LIBRARIES Core)
endif()

configure_file(commands1.txt . COPYONLY)
configure_file(commands2.txt . COPYONLY)


ROOTTEST_ADD_TEST(semaphore
                 MACRO  tSemaphore.C
                 OUTCNVCMD grep -v -e  "tSemaphore"
                 OUTREF tSemaphore.ref)

if(OPENGL_gl_LIBRARY AND ROOTTEST_OS_ID MATCHES Scientific|CentOS|Ubuntu|Fedora)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=${OPENGL_gl_LIBRARY})
endif()

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(fork
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/fork
                    FAILREGEX "Error in" "cannot load any more object with static TLS"
                    DEPENDS ${GENERATE_EXECUTABLE_TEST} roottest-root-multicore-tsenums roottest-root-multicore-tclass_methods)
endif()

if(ROOT_imt_FOUND)
   ROOTTEST_GENERATE_EXECUTABLE(ttree_read_imt ttree_read_imt.cpp LIBRARIES Core Imt Thread Tree RIO)

   ROOTTEST_ADD_TEST(ttree_read_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ttree_read_imt.sh
                     FAILREGEX "ERROR"
                     OUTREF ttree_read_imt.ref
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

### Keep this test for debugging purposes: even if ttree_read_imt makes it redundant, it can be useful to
### debug issues that are hard to reproduce locally but can eventually be observed on the test machines.
#   ROOTTEST_GENERATE_EXECUTABLE(ttree_read_imt_allpar ttree_read_imt_allpar.cpp LIBRARIES Core Thread Tree)

#   ROOTTEST_ADD_TEST(ttree_read_imt_allpar
#                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ttree_read_imt_allpar.sh
#                     FAILREGEX "ERROR"
#                     OUTREF ttree_read_imt_allpar.ref
#                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

   ROOTTEST_GENERATE_EXECUTABLE(ttree_write_imt ttree_write_imt.cpp LIBRARIES Core Imt Thread Tree Hist RIO)

   ROOTTEST_ADD_TEST(ttree_write_imt
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/ttree_write_imt
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

   ROOTTEST_GENERATE_EXECUTABLE(tp_process_imt tp_process_imt.cpp LIBRARIES Core Imt Thread Tree TreePlayer)

   ROOTTEST_ADD_TEST(tp_process_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tp_process_imt.sh
                     COPY_TO_BUILDDIR tp_process_imt.root
                     FAILREGEX "ERROR"
                     OUTREF tp_process_imt.ref
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

   ROOTTEST_GENERATE_EXECUTABLE(tp_process_imt_race_regression tp_process_imt_race_regression.cpp LIBRARIES Core Imt Thread Tree TreePlayer)

   ROOTTEST_ADD_TEST(tp_process_imt_race_regression
                     EXEC ${CMAKE_CURRENT_BINARY_DIR}/tp_process_imt_race_regression
                     COPY_TO_BUILDDIR treeprocmt_race_regression_input1.root treeprocmt_race_regression_input2.root
                                      treeprocmt_race_regression_input3.root treeprocmt_race_regression_input4.root
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()
