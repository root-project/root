if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(ttree
                    MACRO PyROOT_ttreetests.py
                    COPY_TO_BUILDDIR TTreeTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ TTreeTypes.C+
                    ${PYTESTS_WILLFAIL})
endif()
