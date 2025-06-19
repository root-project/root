ROOTTEST_GENERATE_DICTIONARY(libRoottestIoTransientBase
                             testobject.h testobjectderived.h
                             LINKDEF baseLinkDef.h
                             SOURCES testobject.cpp testobjectderived.cpp
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-transient-base-dict-fixture)

ROOTTEST_ADD_TEST(WriteFile
                  COPY_TO_BUILDDIR testobject.h testobjectderived.h
                  MACRO execWriteFile.cxx+
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"libRoottestIoTransientBase\")"
                  OUTREF execWriteFile${ref_suffix}
                  TIMEOUT 600
                  FIXTURES_REQUIRED root-io-transient-base-dict-fixture
                  FIXTURES_SETUP root-io-transient-base-WriteFile-fixture)

if(${compression_default} STREQUAL "lz4")
  set(outref_suffix "LZ4")
else()
  set(outref_suffix "ZLIB")
endif()

ROOTTEST_ADD_TEST(hadd_autoload
                  COMMAND hadd -f data_merge.root data1.root data2.root
                  OUTREF hadd_autoload${outref_suffix}.ref
                  FIXTURES_REQUIRED root-io-transient-base-WriteFile-fixture
                  ENVIRONMENT ROOT_INCLUDE_PATH=${CMAKE_CURRENT_SOURCE_DIR})
