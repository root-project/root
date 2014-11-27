ROOTTEST_GENERATE_EXECUTABLE(fork fork.cpp LIBRARIES Core)

configure_file(commands1.txt . COPYONLY)
configure_file(commands2.txt . COPYONLY)

ROOTTEST_ADD_TEST(roottest-root-multicore-fork
                  EXEC fork
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})
