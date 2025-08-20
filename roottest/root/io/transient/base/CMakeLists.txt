
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ROOTTEST_NOROOTMAP OFF)
ROOT_GENERATE_DICTIONARY(base_cling ${CMAKE_CURRENT_SOURCE_DIR}/base.h MODULE base LINKDEF baseLinkDef.h)
ROOTTEST_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES ROOT::Core ROOT::RIO)

ROOTTEST_ADD_TEST(WriteFile
                  COPY_TO_BUILDDIR testobject.h testobjectderived.h
                  PRECMD ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} ${build_config} --target base${fast}
                  MACRO execWriteFile.cxx+
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"libbase\")"
                  OUTREF execWriteFile${ref_suffix}
                  TIMEOUT 600
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
