############################################################################
# CMakeLists.txt file for building ROOT core/clingutils package
############################################################################

# These files depend on cling/clang/llvm; they need to be linked into libCling.
# They are used by rootcling_stage1, rootcling and libCling.

set(ClingUtils_dict_headers
  root_std_complex.h
  TClingRuntime.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(ClingUtils
  src/RStl.cxx
  src/TClingUtils.cxx
)

add_dependencies(ClingUtils CLING)

target_include_directories(ClingUtils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)
target_include_directories(ClingUtils PRIVATE ${CLING_INCLUDE_DIRS})
# Register the llvm include directories after clangs. This instructs the compiler to resolve
# headers from our builtin clang. That's an issue when we are building with bultin_llvm=Off
# and we have installed clang headers, too.
target_include_directories(ClingUtils PRIVATE SYSTEM ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLING_CXXFLAGS}")

# This is to avoid warnings coming from GCC 7 in llvm/src/include/llvm/ADT/DenseMap.h:1010
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-maybe-uninitialized)
endif()

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
  ROOT_STANDARD_LIBRARY_PACKAGE(${dict}Dict
                                NO_SOURCES NO_INSTALL_HEADERS NO_MODULE
                                STAGE1
                                NODEPHEADERS ${header}
                                LINKDEF src/${dict}Linkdef.h
                                DEPENDENCIES Core)
endforeach()

if(builtin_clang)
  set(CLANG_RESOURCE_DIR_STEM ${CMAKE_BINARY_DIR}/interpreter/llvm/src/${CMAKE_CFG_INTDIR}/lib/clang)
  set(CLANG_RESOURCE_DIR_VERSION ${LLVM_VERSION})
else()
  execute_process(
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -print-resource-dir
    RESULT_VARIABLE _CLANG_ERROR OUTPUT_VARIABLE CLANG_RESOURCE_DIR)

  if(_CLANG_ERROR)
    message(FATAL_ERROR "Cannot determine clang resource directory path")
  endif()

  string(STRIP ${CLANG_RESOURCE_DIR} CLANG_RESOURCE_DIR)
  get_filename_component(CLANG_RESOURCE_DIR ${CLANG_RESOURCE_DIR} REALPATH)
  get_filename_component(CLANG_RESOURCE_DIR_STEM ${CLANG_RESOURCE_DIR} DIRECTORY)
  get_filename_component(CLANG_RESOURCE_DIR_VERSION ${CLANG_RESOURCE_DIR} NAME)
endif()

set(CLANG_RESOURCE_DIR ${CLANG_RESOURCE_DIR_STEM}/${CLANG_RESOURCE_DIR_VERSION}/include)


#---Deal with clang resource here----------------------------------------------
install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include/
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include USE_SOURCE_PERMISSIONS)


#---Install a bunch of files to /etc/cling------------------------------------
set(clinginclude    ${CMAKE_SOURCE_DIR}/interpreter/cling/include)

foreach(file module.modulemap.build
        Interpreter/DynamicExprInfo.h
        Interpreter/DynamicLookupRuntimeUniverse.h
        Interpreter/DynamicLookupLifetimeHandler.h
        Interpreter/Exception.h
        Interpreter/RuntimePrintValue.h
        Interpreter/RuntimeUniverse.h
        Interpreter/Value.h)
  get_filename_component(path ${file} PATH)
  set(dest_file ${file})
  if (${file} STREQUAL "module.modulemap.build")
    set(dest_file "module.modulemap")
  else()
    # We do not want our modulemap to be considered part of the PCH.
    set_property(GLOBAL APPEND PROPERTY CLINGETCPCH etc/cling/${dest_file})
  endif()
  list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy ${clinginclude}/cling/${file} ${CMAKE_BINARY_DIR}/etc/cling/${dest_file})
  list(APPEND files_to_copy ${clinginclude}/cling/${file})
  install(FILES ${CMAKE_BINARY_DIR}/etc/cling/${dest_file} DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/cling/${path})
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
    list(APPEND copy_commands COMMAND ${CMAKE_COMMAND} -E copy /usr/include/${file} ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include/${file})
  endif()
endforeach()

set(stamp_file ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/LLVMRES.stamp)
if(MSVC)
  add_custom_command(OUTPUT ${stamp_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include
        ${copy_commands}
        COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
        DEPENDS ${files_to_copy}
        COMMENT "Copying LLVM resource and header files")
else()
  add_custom_command(OUTPUT ${stamp_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CLANG_RESOURCE_DIR}
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/assert.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include/assert.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/stdlib.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include/stdlib.h
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/interpreter/llvm/ROOT/unistd.h
        ${CMAKE_BINARY_DIR}/etc/cling/lib/clang/${CLANG_RESOURCE_DIR_VERSION}/include/unistd.h
        ${copy_commands}
        COMMAND ${CMAKE_COMMAND} -E touch ${stamp_file}
        DEPENDS ${files_to_copy}
        COMMENT "Copying LLVM resource and header files")
endif()
add_custom_target(LLVMRES DEPENDS ${stamp_file} CLING)

