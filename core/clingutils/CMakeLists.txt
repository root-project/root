############################################################################
# CMakeLists.txt file for building ROOT core/clingutils package
############################################################################

# These files depend on cling/clang/llvm; they need to be linked into libCling.
# They are used by rootcling_stage1, rootcling and libCling.

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

set(ClingUtils_dict_headers ${headers} PARENT_SCOPE)

include_directories(${LLVM_INCLUDE_DIRS} ${CLING_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLING_CXXFLAGS}")

# This is to avoid warnings coming from GCC 7 in llvm/src/include/llvm/ADT/DenseMap.h:1010
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-maybe-uninitialized)
endif()

ROOT_OBJECT_LIBRARY(ClingUtils ${sources})

add_dependencies(ClingUtils CLING)

ROOT_INSTALL_HEADERS()

#### STL dictionary (replacement for cintdlls)##############################

set(stldicts
    vector
    list
    forward_list
    deque
    map map2 unordered_map
    multimap multimap2 unordered_multimap
    set unordered_set
    multiset unordered_multiset
    complex)
if(NOT WIN32)
  list(APPEND stldicts valarray)
endif()
include_directories(${CMAKE_BINARY_DIR}/etc/cling/cint)
foreach(dict ${stldicts})
  string(REPLACE "2" "" header ${dict})
  string(REPLACE "complex" "root_std_complex.h" header ${header})
  string(REPLACE "multi" "" header ${header})
  ROOT_GENERATE_DICTIONARY(G__${dict}Dict ${header} STAGE1 MODULE ${dict}Dict LINKDEF src/${dict}Linkdef.h DEPENDENCIES  Core)
  ROOT_LINKER_LIBRARY(${dict}Dict G__${dict}Dict.cxx DEPENDENCIES  Core)
endforeach()

#---Deal with LLVM resource here----------------------------------------------
set(copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include/llvm/Config/llvm-config.h
        ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config/llvm-config.h)
set(files_to_copy ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include/llvm/Config/llvm-config.h)

install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/llvm/Config DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/llvm)
install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/lib/clang/${LLVM_VERSION}/include USE_SOURCE_PERMISSIONS)


#---Install a bunch of files to /etc/cling------------------------------------
set(clinginclude    ${CMAKE_SOURCE_DIR}/interpreter/cling/include)
set(llvminclude     ${CMAKE_SOURCE_DIR}/interpreter/llvm/src/include)
set(llvminclude-bin ${CMAKE_BINARY_DIR}/interpreter/llvm/src/include)

foreach(file  Interpreter/DynamicExprInfo.h
        Interpreter/DynamicLookupRuntimeUniverse.h
        Interpreter/DynamicLookupLifetimeHandler.h
        Interpreter/Exception.h
        Interpreter/RuntimePrintValue.h
        Interpreter/RuntimeUniverse.h
        Interpreter/Value.h)
  get_filename_component(path ${file} PATH)
  list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${clinginclude}/cling/${file} ${CMAKE_BINARY_DIR}/etc/cling/${file})
  list(APPEND files_to_copy ${clinginclude}/cling/${file})
  set_property(GLOBAL APPEND PROPERTY CLINGETCPCH etc/cling/${file})
  install(FILES ${clinginclude}/cling/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
endforeach()

foreach(file  multimap  multiset)
  list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} ${CMAKE_BINARY_DIR}/etc/cling/cint/${file})
  list(APPEND files_to_copy ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file})
  install(FILES ${CMAKE_SOURCE_DIR}/interpreter/cling/include/cling/cint/${file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/cint)
endforeach()

# These headers do not have complete include guards on all platforms we
# support. This means that the PCH cannot provide their representation at
# runtime and clang will hit disk, triggering a possible incompatibility
# of that file in build-time versus run-time (different length etc).
# Capture their build-time version here, and inject it into runtime.
foreach(file wchar.h bits/stat.h bits/time.h)
  if(EXISTS /usr/include/${file})
    list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy /usr/include/${file} ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/${file})
  endif()
endforeach()

set(stamp_file ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LLVMRES.stamp)
add_custom_command(OUTPUT ${stamp_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/interpreter/llvm/src/${CMAKE_CFG_INTDIR}/lib/clang/${LLVM_VERSION}/include
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/assert.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/assert.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/stdlib.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/stdlib.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/unistd.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${LLVM_VERSION}/include/unistd.h
        ${copy_commands}
        COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
        DEPENDS ${files_to_copy}
        COMMENT "Copying LLVM resource and header files")
add_custom_target(LLVMRES DEPENDS ${stamp_file} CLING)

