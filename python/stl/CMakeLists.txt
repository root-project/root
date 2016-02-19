if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(stl
                    MACRO PyROOT_stltests.py
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ StlTypes.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    # OUTREF PyROOT_stltests.ref
                    )
endif()
