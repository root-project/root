if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(stl
                    MACRO PyROOT_stltests.py
                    COPY_TO_BUILDDIR StlTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ StlTypes.C+
                    ${PYTESTS_WILLFAIL})
endif()

