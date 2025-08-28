if(ROOT_pyroot_FOUND)
  set(scott_file ScottCppyy.C) # Uses Cppyy header

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(regression
                    MACRO PyROOT_regressiontests.py OPTS -b
                    LABELS longtest
                    COPY_TO_BUILDDIR ${scott_file} Scott2.C Scott3.C MuonTileID.C LoKiNamespace.C ULongLong.C
                                     Till.C CoralAttributeList.C NestedNamespace.h Marco.C
                                     PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->Load\(\"libROOTTPython\"\)
                                         -e .L\ ${scott_file}+
                                         -e .L\ Scott2.C+
                                         -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ LoKiNamespace.C+
                                         -e .L\ ULongLong.C+
                                         -e .L\ Till.C+
                                         -e .L\ CoralAttributeList.C+)
endif()

  ROOTTEST_ADD_TEST(root_6023
                    MACRO exec_root_6023.py
                    COPY_TO_BUILDDIR root_6023.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ root_6023.h+)

  ROOTTEST_ADD_TEST(gh_16406
                    MACRO gh_16406.py
                    OUTREF gh_16406.ref)
endif()
