if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-regression MACRO PyROOT_regressiontests.py OPTS -b
                    PRECMD root -b -q -l -e .L\ Scott.C+ 
                                         -e .L\ Scott2.C+
                                         -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ LoKiNamespace.C+
                                         -e .L\ ULongLong.C+
                                         -e .L\ Till.C+
                                         -e .L\ CoralAttributeList.C+)
endif()
