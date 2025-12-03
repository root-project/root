if(NOT geom)
  return()
endif()

if(gdml)
  set(_ref execParse.ref)
else()
  set(_ref execParse.ref-nogdml)
endif()

ROOTTEST_ADD_TEST(Parse
                  MACRO execParse.cxx+
                  OUTREF ${ref}
                  COPY_TO_BUILDDIR dune10kt_v1_workspace.gdml)

ROOTTEST_ADD_TEST(unpack
                  COMMAND tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/geometries.tar.gz
                  FIXTURES_SETUP root-io-geo-unpack-fixture)

foreach(macro aleph_all alice_all ams_all babar_all brahms_all btev_all cdf_all cms_cmse d0_all e907_all gem_all hades_all lhcb_mag lhcbfull_all phenix_all phobos_all star_all tesla_all)
  ROOTTEST_ADD_TEST(export_${macro}
                    MACRO runexport.C
                    MACROARG "\"${CMAKE_CURRENT_BINARY_DIR}/${macro}\""
                    FIXTURES_REQUIRED root-io-geo-unpack-fixture)
endforeach()

ROOTTEST_ADD_TEST(readGeo
                  MACRO runreadGeo.C
                  OUTREF readGeo.ref
                  FIXTURES_REQUIRED root-io-geo-unpack-fixture)
