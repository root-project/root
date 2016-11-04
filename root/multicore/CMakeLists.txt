ROOTTEST_ADD_TESTDIRS()

# Workaround (k)ubuntu bug: runtime issue when using pthread and gcc48
if(ROOTTEST_OS_ID MATCHES Ubuntu)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

ROOTTEST_GENERATE_EXECUTABLE(processExecutorTreeTest tProcessExecutorTreeTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

ROOTTEST_ADD_TEST(processExecutorTreeTest
                  EXEC processExecutorTreeTest
                  OUTREF tProcessExecutorTreeTest.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(processExecutorProcessTest tProcessExecutorProcessTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

ROOTTEST_ADD_TEST(processExecutorProcess
                  EXEC processExecutorProcessTest
                  OUTREF tProcessExecutorProcessTest.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(processExecutor tProcessExecutor.cpp LIBRARIES MultiProc Core Net TreePlayer RIO Hist)

ROOTTEST_ADD_TEST(processExecutor
                  EXEC processExecutor
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

if(ROOT_imt_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  ROOTTEST_GENERATE_EXECUTABLE(threadExecutor tThreadExecutor.cpp LIBRARIES MultiProc Core Net TreePlayer RIO Hist ${TBB_LIBRARIES})

ROOTTEST_ADD_TEST(threadExecutor
                    EXEC threadExecutor
                    DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()
                  
ROOTTEST_GENERATE_EXECUTABLE(current_dir current_dir.cpp LIBRARIES Core RIO Thread)

ROOTTEST_ADD_TEST(current_dir
                  EXEC current_dir
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

ROOTTEST_GENERATE_EXECUTABLE(tclass_methods tclass_methods.cpp LIBRARIES Core Thread)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tclass_methods tclass_methods.h  SELECTION tclass_methods_selection.xml)

ROOTTEST_ADD_TEST(tclass_methods
                  EXEC tclass_methods
                  DEPENDS ${GENERATE_EXECUTABLE_TEST} ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_EXECUTABLE(ttree_write ttree_write.cpp LIBRARIES Core Hist RIO Thread Tree)

ROOTTEST_ADD_TEST(ttree_write
                  EXEC ttree_write
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(ttree_read ttree_read.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(ttree_read.root . COPYONLY)

ROOTTEST_ADD_TEST(ttree_read
                  EXEC ttree_read
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(th1f_fill th1f_fill.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(file1.root . COPYONLY)
configure_file(file2.root . COPYONLY)
configure_file(file3.root . COPYONLY)

ROOTTEST_ADD_TEST(th1f_fill
                  EXEC th1f_fill
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# ROOTTEST_GENERATE_EXECUTABLE(tformula tformula.cpp LIBRARIES Core Hist Thread)
#
# ROOTTEST_ADD_TEST(tformula
#                   EXEC tformula
#                   DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(tprofile tprofile.cpp LIBRARIES Core Hist Thread)

ROOTTEST_ADD_TEST(tprofile
                  EXEC tprofile
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# Re-Enabled Disable test: see ROOT-7158
ROOTTEST_GENERATE_REFLEX_DICTIONARY(tsenums tsenums.h  SELECTION tsenums_selection.xml)

ROOTTEST_ADD_TEST(tsenums
                  MACRO  exectsenums.C
                  OUTREF exectsenums.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_EXECUTABLE(fork fork.cpp LIBRARIES Core)

configure_file(commands1.txt . COPYONLY)
configure_file(commands2.txt . COPYONLY)


ROOTTEST_ADD_TEST(semaphore
                 MACRO  tSemaphore.C
                 OUTCNVCMD grep -v -e  "tSemaphore"
                 OUTREF tSemaphore.ref)

if(ROOTTEST_OS_ID MATCHES Scientific OR ROOTTEST_OS_ID MATCHES CentOS)
  set(ROOTTEST_ENV_EXTRA LD_PRELOAD=${OPENGL_gl_LIBRARY})
endif()

ROOTTEST_ADD_TEST(fork
                  EXEC fork
                  FAILREGEX "Error in" "cannot load any more object with static TLS"
                  DEPENDS ${GENERATE_EXECUTABLE_TEST} tsenums tclass_methods)

if(ROOT_imt_FOUND)
   ROOTTEST_ADD_TEST(generate_imt_tree
                     MACRO generate_imt_tree.C+)

   ROOTTEST_GENERATE_EXECUTABLE(ttree_read_imt ttree_read_imt.cpp LIBRARIES Core Thread Tree)
   ROOTTEST_GENERATE_EXECUTABLE(ttree_write_imt ttree_write_imt.cpp LIBRARIES Core Thread Tree Hist)

   ROOTTEST_ADD_TEST(ttree_read_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ttree_read_imt.sh
                     FAILREGEX "ERROR"
                     OUTREF ttree_read_imt.ref
                     DEPENDS generate_imt_tree ${GENERATE_EXECUTABLE_TEST})

   ROOTTEST_ADD_TEST(ttree_write_imt
                     EXEC ttree_write_imt
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})

   ROOTTEST_GENERATE_EXECUTABLE(tp_process_imt tp_process_imt.cpp LIBRARIES Core Thread Tree TreePlayer)

   ROOTTEST_ADD_TEST(tp_process_imt
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tp_process_imt.sh
                     FAILREGEX "ERROR"
                     OUTREF tp_process_imt.ref
                     DEPENDS ${GENERATE_EXECUTABLE_TEST})
   
endif()
