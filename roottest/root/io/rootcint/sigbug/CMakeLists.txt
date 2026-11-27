set(CMAKE_ROOTTEST_DICT ON)
set(CMAKE_ROOTTEST_NOROOTMAP ON)

set(dictname Midad)

ROOT_GENERATE_DICTIONARY(${dictname}
                         ${CMAKE_CURRENT_SOURCE_DIR}/Mint.h
                         NO_CXXMODULE
                         MODULE ${dictname}
                         LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h
                         OPTIONS -DUNIX -DLINUX -D__UNIX__ -D__LINUX__ -DDATAREP_LITTLE_IEEE -DDATAREP_LITTLE_ENDIAN -DDEFECT_NO_IOSTREAM_NAMESPACES -DDEFECT_NO_JZEXT -DDEFECT_NO_INTHEX -DDEFECT_NO_INTHOLLERITH -DDEFECT_NO_READONLY -DDEFECT_NO_DIRECT_FIXED -DDEFECT_NO_STRUCTURE -DUSE_ROOT)

# Custom target to trigger the dictionary generation test at build time, which is easier
# and more robust than the previous add_test(.. COMMAND ${CMAKE_COMMAND} --build ..) pattern.
add_custom_target(${dictname}Target ALL)
add_dependencies(${dictname}Target ${dictname})
# With MSVC, roottest dictionaries don't depend on rootcling (see ROOT_GENERATE_DICTIONARY and 5d86d90d487)
add_dependencies(${dictname} rootcling)
