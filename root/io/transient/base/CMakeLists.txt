
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ROOTTEST_NOROOTMAP OFF)
ROOT_GENERATE_DICTIONARY(base_cling ${CMAKE_CURRENT_SOURCE_DIR}/base.h MODULE base LINKDEF baseLinkDef.h)
ROOTTEST_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES Core RIO)

set(RootExeOptions -e "gSystem->Load(\"libbase\")")   # Adding a ; to avoid the output is complicated

ROOTTEST_ADD_TEST(WriteFile
                  COPY_TO_BUILDDIR testobject.h testobjectderived.h
                  PRECMD ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target base${fast}
                  MACRO execWriteFile.cxx+
                  OUTREF execWriteFile${ref_suffix}
                  TIMEOUT 600)

if(${compression_default} STREQUAL "lz4")
  set(outref_suffix "LZ4")
else()
  set(outref_suffix "ZLIB")
endif()
ROOTTEST_ADD_TEST(hadd_autoload
  COMMAND hadd -f data_merge.root data1.root data2.root
  OUTREF hadd_autoload${outref_suffix}.ref
  DEPENDS WriteFile
  ENVIRONMENT ROOT_INCLUDE_PATH=${CMAKE_CURRENT_SOURCE_DIR})
