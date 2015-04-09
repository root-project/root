ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_EXECUTABLE(current_dir current_dir.cpp LIBRARIES Core RIO Thread)

ROOTTEST_ADD_TEST(roottest-root-multicore-current_dir
                  EXEC current_dir
                  OUTREF current_dir.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_ADD_TEST(roottest-root-multicore-exectstformula
                  MACRO exectstformula.C
                  OUTREF exectstformula.ref)

ROOTTEST_ADD_TEST(roottest-root-multicore-exectsautoparse
                  MACRO exectsautoparse.C
                  OUTREF exectsautoparse.ref)

ROOTTEST_ADD_TEST(roottest-root-multicore-exectsinclude
                 MACRO  exectsinclude.C
                 OUTCNVCMD grep -v -e "Line processed"
                 OUTREF exectsinclude.ref)

ROOTTEST_ADD_TEST(roottest-root-multicore-execautoloadts
                 MACRO  execautoloadts.C
                 OUTREF execautoloadts.ref)

ROOTTEST_GENERATE_EXECUTABLE(tclass_methods tclass_methods.cpp LIBRARIES Core Thread)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tclass_methods tclass_methods.h  SELECTION tclass_methods_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-multicore-tclass_methods
                  EXEC tclass_methods
                  DEPENDS ${GENERATE_EXECUTABLE_TEST} ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_EXECUTABLE(ttree_write ttree_write.cpp LIBRARIES Core Hist RIO Thread Tree)

ROOTTEST_ADD_TEST(roottest-root-multicore-ttree_write
                  EXEC ttree_write
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(ttree_read ttree_read.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(ttree_read.root . COPYONLY)

ROOTTEST_ADD_TEST(roottest-root-multicore-ttree_read
                  EXEC ttree_read
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(th1f_fill th1f_fill.cpp LIBRARIES Core Hist RIO Thread Tree)

configure_file(file1.root . COPYONLY)
configure_file(file2.root . COPYONLY)
configure_file(file3.root . COPYONLY)

ROOTTEST_ADD_TEST(roottest-root-multicore-th1f_fill
                  EXEC th1f_fill
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# ROOTTEST_GENERATE_EXECUTABLE(tformula tformula.cpp LIBRARIES Core Hist Thread)
#
# ROOTTEST_ADD_TEST(roottest-root-multicore-tformula
#                   EXEC tformula
#                   DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_GENERATE_EXECUTABLE(tprofile tprofile.cpp LIBRARIES Core Hist Thread)

ROOTTEST_ADD_TEST(roottest-root-multicore-tprofile
                  EXEC tprofile
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

# Disable test: see ROOT-7158
#ROOTTEST_GENERATE_REFLEX_DICTIONARY(tsenums tsenums.h  SELECTION tsenums_selection.xml)

#ROOTTEST_ADD_TEST(roottest-root-multicore-tsenums
#                  MACRO  exectsenums.C
#                  OUTREF exectsenums.ref
#                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_EXECUTABLE(fork fork.cpp LIBRARIES Core)

configure_file(commands1.txt . COPYONLY)
configure_file(commands2.txt . COPYONLY)

ROOTTEST_ADD_TEST(roottest-root-multicore-fork
                  EXEC fork
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})
