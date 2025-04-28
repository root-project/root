if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(ttree
                    MACRO PyROOT_ttreetests.py
                    COPY_TO_BUILDDIR TTreeTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ TTreeTypes.C+)
endif()
