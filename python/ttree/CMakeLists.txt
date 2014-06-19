if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-ttree
                    MACRO PyROOT_ttreetests.py
                    PRECMD root -b -q -l -e .L\ TTreeTypes.C+
                    OUTREF PyROOT_ttreetests.ref)
endif()
