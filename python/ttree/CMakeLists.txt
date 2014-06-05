if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-ttree MACRO PyROOT_ttreetests.py
                    PRECMD root -b -q -l -e .L\ TTreeTypes.C+)
endif()
