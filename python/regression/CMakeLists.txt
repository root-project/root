if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(regression
                    MACRO PyROOT_regressiontests.py OPTS -b
                    LABELS longtest
                    COPY_TO_BUILDDIR Scott.C Scott2.C Scott3.C MuonTileID.C LoKiNamespace.C ULongLong.C 
                                     Till.C CoralAttributeList.C NestedNamespace.h Marco.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARY}\"\)
                                         -e .L\ Scott.C+
                                         -e .L\ Scott2.C+
                                         -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ LoKiNamespace.C+
                                         -e .L\ ULongLong.C+
                                         -e .L\ Till.C+
                                         -e .L\ CoralAttributeList.C+
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(root_6023
                    MACRO exec_root_6023.py OPTS -b
                    COPY_TO_BUILDDIR root_6023.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARY}\"\)
                                                     -e .L\ root_6023.h+
                    ${PYTESTS_WILLFAIL})

endif()
