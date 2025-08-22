include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Edm
                                    inc/Wrapper.h
                                    SELECTION inc/edm_selection.xml
                                    FIXTURES_SETUP root-meta-cmsUnload-Edm-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Strip
                                    inc/SiStripCluster_classes.h
                                    SELECTION inc/strip_selection.xml
                                    FIXTURES_SETUP root-meta-cmsUnload-Strip-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Cluster
                                    inc/ClusterTotals.h
                                    SELECTION inc/cluster_selection.xml
                                    FIXTURES_SETUP root-meta-cmsUnload-Cluster-fixture)


macro(CMS_UNLOAD_TEST name use_rmap)
  if(${use_rmap} STREQUAL "true")
    set(testname rmap-${name})
  else()
    set(testname ${name})
  endif()

  ROOTTEST_ADD_TEST(${testname}
                    MACRO runscript.C
                    COPY_TO_BUILDDIR step2.cxx
                    MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}/scripts/${name}.script\",${use_rmap}"
                    ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                    OUTREF ${testname}.ref
                    FIXTURES_REQUIRED root-meta-cmsUnload-Edm-fixture
                                      root-meta-cmsUnload-Strip-fixture
                                      root-meta-cmsUnload-Cluster-fixture)
endmacro(CMS_UNLOAD_TEST)

CMS_UNLOAD_TEST(case1 false)
CMS_UNLOAD_TEST(case2 false)
CMS_UNLOAD_TEST(case3 false)

CMS_UNLOAD_TEST(case1 true)
CMS_UNLOAD_TEST(case2 true)
CMS_UNLOAD_TEST(case3 true)
