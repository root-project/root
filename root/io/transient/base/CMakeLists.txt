
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ROOTTEST_NOROOTMAP OFF)
ROOT_GENERATE_DICTIONARY(base_cling ${CMAKE_CURRENT_SOURCE_DIR}/base.h MODULE base LINKDEF baseLinkDef.h)
if(MSVC)
    ROOT_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES libCore libRIO)
    add_custom_command(TARGET base POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libbase.dll
                                         ${CMAKE_CURRENT_BINARY_DIR}/libbase.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libbase.lib
                                         ${CMAKE_CURRENT_BINARY_DIR}/libbase.lib)
    set(writefile_outref execWriteFile_win32.ref)
else()
	ROOT_LINKER_LIBRARY(base base_cling.cxx testobject.cpp testobjectderived.cpp TEST LIBRARIES Core RIO)
    set(writefile_outref execWriteFile.ref)
endif()

set(RootExeOptions -e "gSystem->Load(\"libbase\")")   # Adding a ; to avoid the output is complicated

ROOTTEST_ADD_TEST(WriteFile
                  PRECMD ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target base${fast}
                  MACRO execWriteFile.cxx+
                  OUTREF ${writefile_outref})
if(${compression_default} STREQUAL "lz4")
	ROOTTEST_ADD_TEST(hadd_autoload
                  	COMMAND hadd -f data_merge.root data1.root data2.root
                  	OUTREF hadd_autoloadLZ4.ref
                  	DEPENDS WriteFile)
else()
	ROOTTEST_ADD_TEST(hadd_autoload
                  	COMMAND hadd -f data_merge.root data1.root data2.root
                  	OUTREF hadd_autoloadZLIB.ref
                  	DEPENDS WriteFile)
endif()
