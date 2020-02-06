if(ROOT_pyroot_FOUND)
  if (ROOT_pyroot_experimental_FOUND)
    set(scott_file ScottCppyy.C) # Uses Cppyy header
  else()
    set(scott_file Scott.C)
  endif()

  ROOTTEST_ADD_TEST(regression
                    MACRO PyROOT_regressiontests.py OPTS -b
                    LABELS longtest
                    COPY_TO_BUILDDIR ${scott_file} Scott2.C Scott3.C MuonTileID.C LoKiNamespace.C ULongLong.C
                                     Till.C CoralAttributeList.C NestedNamespace.h Marco.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARIES}\"\)
                                         -e .L\ ${scott_file}+
                                         -e .L\ Scott2.C+
                                         -e .L\ Scott3.C+
                                         -e .L\ MuonTileID.C+
                                         -e .L\ LoKiNamespace.C+
                                         -e .L\ ULongLong.C+
                                         -e .L\ Till.C+
                                         -e .L\ CoralAttributeList.C+
                    ENVIRONMENT EXP_PYROOT=${exp_pyroot})

  ROOTTEST_ADD_TEST(root_6023
                    MACRO exec_root_6023.py OPTS -b
                    COPY_TO_BUILDDIR root_6023.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e gSystem->AddLinkedLibs\(\"${PYTHON_LIBRARIES}\"\)
                                                     -e .L\ root_6023.h+
                    ENVIRONMENT CLING_STANDARD_PCH=none
                    CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/${python_version}/libcppyy_backend.so)

endif()
