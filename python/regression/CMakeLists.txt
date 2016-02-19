if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(regression
                    MACRO PyROOT_regressiontests.py OPTS -b
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARY}\"\)
                                         -e .L\ Scott.C+
                                         -e .L\ Scott2.C+
                                         -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ LoKiNamespace.C+
                                         -e .L\ ULongLong.C+
                                         -e .L\ Till.C+
                                         -e .L\ CoralAttributeList.C+
                    # OUTREF PyROOT_regressiontests.ref
                    )
  ROOTTEST_ADD_TEST(root_6023
                    MACRO exec_root_6023.py OPTS -b
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARY}\"\)
                                         -e .L\ root_6023.h+
                    )

endif()
