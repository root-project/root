ROOTTEST_ADD_TESTDIRS()

# Workaround (k)ubuntu bug: runtime issue when using pthread and gcc48
if(UNIX)
EXEC_PROGRAM(uname
             ARGS -v
             OUTPUT_VARIABLE LINUX_VERSION)
STRING(FIND LINUX_VERSION
       "buntu"
       BUNTU_MATCH)
if(NOT BUNTU_MATCH STREQUAL "")
   set(UBUNTU_EXTRA_FLAGS "-Wl,--no-as-needed")
endif()

endif()




ROOTTEST_GENERATE_EXECUTABLE(procPoolTreeTest tProcPoolTreeTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

ROOTTEST_ADD_TEST(procPoolTreeTest
                  EXEC procPoolTreeTest
                  OUTREF tProcPoolTreeTest.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(procPoolProcessTest tProcPoolProcessTest.cpp LIBRARIES MultiProc Core Net TreePlayer Tree RIO Hist)

ROOTTEST_ADD_TEST(procPoolProcess
                  EXEC procPoolProcessTest
                  OUTREF tProcPoolProcessTest.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(procPool tprocpool.cpp LIBRARIES MultiProc Core Net TreePlayer RIO Hist)

ROOTTEST_ADD_TEST(procPool
                  EXEC procPool
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

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

ROOTTEST_GENERATE_EXECUTABLE(tclass_methods tclass_methods.cpp LIBRARIES Core Thread COMPILE_FLAGS ${UBUNTU_EXTRA_FLAGS})

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

ROOTTEST_GENERATE_EXECUTABLE(tprofile tprofile.cpp LIBRARIES Core Hist Thread COMPILE_FLAGS ${UBUNTU_EXTRA_FLAGS})

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

ROOTTEST_ADD_TEST(fork
                  EXEC fork
                  FAILREGEX "Error in"
                  DEPENDS ${GENERATE_EXECUTABLE_TEST} tsenums tclass_methods)

ROOTTEST_ADD_TEST(semaphore
                 MACRO  tSemaphore.C
                 OUTCNVCMD grep -v -e  "tSemaphore"
                 OUTREF tSemaphore.ref)
