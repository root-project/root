set(CMAKE_ROOTTEST_DICT ON)
set(CMAKE_ROOTTEST_NOROOTMAP ON)

set(dictname Midad)

ROOTTEST_TARGETNAME_FROM_FILE(GENERATE_DICTIONARY_TEST ${dictname})

set(GENERATE_DICTIONARY_TEST ${GENERATE_DICTIONARY_TEST}-build)

ROOT_GENERATE_DICTIONARY(${dictname}
                         ${CMAKE_CURRENT_SOURCE_DIR}/Mint.h
                         NO_CXXMODULE
                         MODULE ${dictname}
                         LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h
                         OPTIONS -DUNIX -DLINUX -D__UNIX__ -D__LINUX__ -DDATAREP_LITTLE_IEEE -DDATAREP_LITTLE_ENDIAN -DDEFECT_NO_IOSTREAM_NAMESPACES -DDEFECT_NO_JZEXT -DDEFECT_NO_INTHEX -DDEFECT_NO_INTHOLLERITH -DDEFECT_NO_READONLY -DDEFECT_NO_DIRECT_FIXED -DDEFECT_NO_STRUCTURE -DUSE_ROOT)

add_test(NAME ${GENERATE_DICTIONARY_TEST}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                    ${build_config}
                                    --target ${dictname}${fast}
                                    -- ${always-make})
