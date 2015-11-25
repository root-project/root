if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(write
                    MACRO PyROOT_writetests.py
                    PRECMD root -b -q -l -e .L\ PickleTypes.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_writetests.ref)

  ROOTTEST_ADD_TEST(read
                    MACRO PyROOT_readtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_readtests.ref
                    DEPENDS roottest-python-pickle-write)
endif()
