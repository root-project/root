
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
ROOT_GENERATE_DICTIONARY(base_cling base.h MODULE base LINKDEF baseLinkDef.h)
ROOT_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES Core RIO)

set(RootExeOptions -e "gSystem->Load(\"libbase\")")   # Adding a ; to avoid the output is complicated

ROOTTEST_ADD_TEST(roottest-root-io-transient-base-WriteFile
                  PRECMD ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target base${fast}
                  MACRO execWriteFile.cxx+
                  OUTREF execWriteFile.ref)

ROOTTEST_ADD_TEST(roottest-root-io-transient-base-hadd_autoload
                  COMMAND hadd -f data_merge.root data1.root data2.root
                  OUTREF hadd_autoload.ref
                  DEPENDS roottest-root-io-transient-base-WriteFile)
