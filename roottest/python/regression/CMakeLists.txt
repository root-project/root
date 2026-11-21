if(pyroot)
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(regression
                    MACRO PyROOT_regressiontests.py OPTS -b
                    LABELS longtest
                    COPY_TO_BUILDDIR Scott3.C MuonTileID.C
                                     CoralAttributeList.C Marco.C
                                     PRECMD ${ROOT_root_CMD} -b -q -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ CoralAttributeList.C+)
endif()

  ROOTTEST_ADD_TEST(root_6023
                    MACRO exec_root_6023.py
                    COPY_TO_BUILDDIR root_6023.h
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ root_6023.h+)

  ROOTTEST_ADD_TEST(gh_16406
                    MACRO gh_16406.py
                    OUTREF gh_16406.ref)

  ROOTTEST_ADD_TEST(gh_21536
                    MACRO gh_21536.py)

  ROOTTEST_GENERATE_REFLEX_DICTIONARY(gh_24176
                                      gh_24176.h
                                      SELECTION gh_24176_selection.xml
                                      FIXTURES_SETUP python-regression-gh_24176-dict-fixture)

  ROOTTEST_ADD_TEST(gh_24176_enum
                    MACRO gh_24176.py
                    OPTS enum
                    FIXTURES_REQUIRED python-regression-gh_24176-dict-fixture)

  ROOTTEST_ADD_TEST(gh_24176_variable
                    MACRO gh_24176.py
                    OPTS variable
                    FIXTURES_REQUIRED python-regression-gh_24176-dict-fixture)
  
  ROOTTEST_ADD_TEST(gh_16613
                    MACRO gh_16613.py)
endif()
