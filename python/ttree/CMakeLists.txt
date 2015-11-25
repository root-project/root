if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(ttree
                    MACRO PyROOT_ttreetests.py
                    PRECMD root -b -q -l -e .L\ TTreeTypes.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_ttreetests.ref)
endif()
