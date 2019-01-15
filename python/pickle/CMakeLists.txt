if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(write
                    MACRO PyROOT_writetests.py
                    COPY_TO_BUILDDIR PickleTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ PickleTypes.C+
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(read
                    MACRO PyROOT_readtests.py
                    DEPENDS write
                    ${PYTESTS_WILLFAIL})
endif()
