if(pyroot)
  ROOTTEST_ADD_TEST(ttree
                    MACRO PyROOT_ttreetests.py
                    COPY_TO_BUILDDIR TTreeTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ TTreeTypes.C+)
endif()
