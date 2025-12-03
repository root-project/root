if(MSVC)
  set(command ${CMAKE_COMMAND} -E env "ROOTIGNOREPREFIX=1" ${CMAKE_BINARY_DIR}/bin/rootcling.exe)
else()
  set(command ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
                                      "ROOTIGNOREPREFIX=1" ${CMAKE_BINARY_DIR}/bin/rootcling)
endif()

ROOTTEST_ADD_TEST(HeaderDict
                  COMMAND ${command} -v2 -f HeaderDict.cxx  ${CMAKE_CURRENT_SOURCE_DIR}/Header.h ${CMAKE_CURRENT_SOURCE_DIR}/Linkdef.h)

ROOTTEST_ADD_TEST(SpaceTeleDict
                  COMMAND ${command} -v2 -f SpaceTeleDict.cxx  ${CMAKE_CURRENT_SOURCE_DIR}/Tele.h ${CMAKE_CURRENT_SOURCE_DIR}/TeleLinkdef.h
                  PASSRC 1
                  ERRREF dicto.ref)
