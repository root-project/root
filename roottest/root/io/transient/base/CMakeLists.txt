
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ROOTTEST_NOROOTMAP OFF)
ROOT_GENERATE_DICTIONARY(base_cling ${CMAKE_CURRENT_SOURCE_DIR}/base.h MODULE base LINKDEF baseLinkDef.h)
ROOTTEST_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES ROOT::Core ROOT::RIO)

add_test(NAME roottest-root-io-transient-base-build
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                  ${build_config}
                                  --target base${fast}
                                  -- ${always-make})
set_property(TEST roottest-root-io-transient-base-build PROPERTY FIXTURES_SETUP root-io-transient-base-build)
if(CMAKE_GENERATOR MATCHES Ninja)
   set_property(TEST roottest-root-io-transient-base-build APPEND PROPERTY RESOURCE_LOCK NINJA_BUILD)
   set_property(TEST roottest-root-io-transient-base-build APPEND PROPERTY FIXTURES_REQUIRED NINJA_BUILD_ALL)
endif()

ROOTTEST_ADD_TEST(WriteFile
                  COPY_TO_BUILDDIR testobject.h testobjectderived.h
                  MACRO execWriteFile.cxx+
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"libbase\")"
                  OUTREF execWriteFile${ref_suffix}
                  TIMEOUT 600
                  FIXTURES_REQUIRED root-io-transient-base-build
                  FIXTURES_SETUP root-io-transient-base-WriteFile-fixture)

if(${compression_default} STREQUAL "lz4")
  set(outref_suffix "LZ4")
else()
  set(outref_suffix "ZLIB")
endif()

set(root_includepaths
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DEFAULT_ROOT_INCLUDE_PATH}
)
cmake_path(CONVERT "${root_includepaths}" TO_NATIVE_PATH_LIST root_includepaths_native)
# Like in tutorials/CMakeLists.txt, we use this generator expression to make
# sure the semicolon in Windows path separators doesn't get used up for list
# parsing at configuration time in ROOTTEST_ADD_TEST.
string(REPLACE ";" "$<SEMICOLON>" root_includepaths_escaped "${root_includepaths_native}")

ROOTTEST_ADD_TEST(hadd_autoload
                  COMMAND hadd -f data_merge.root data1.root data2.root
                  OUTREF hadd_autoload${outref_suffix}.ref
                  FIXTURES_REQUIRED root-io-transient-base-WriteFile-fixture
                  ENVIRONMENT ROOT_INCLUDE_PATH=${root_includepaths_escaped})
