cmake_minimum_required(VERSION 3.13)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

enable_language(CXX)
set(CMAKE_CXX_EXTENSIONS NO)

option(CPPINTEROP_USE_CLING "Use Cling as backend" OFF)
option(CPPINTEROP_USE_REPL "Use clang-repl as backend" ON)
option(CPPINTEROP_ENABLE_TESTING "Enable the CppInterOp testing infrastructure." ON)

if (CPPINTEROP_USE_CLING AND CPPINTEROP_USE_REPL)
message(FATAL_ERROR "We can only use Cling (${CPPINTEROP_USE_CLING}=On) or Repl (CPPINTEROP_USE_REPL=On), but not both of them.")
endif()
# If we are not building as a part of LLVM, build CppInterOp as a standalone
# project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(CppInterOp)

  # LLVM/Clang/Cling default paths
  if (DEFINED LLVM_DIR)
    if (NOT DEFINED Clang_DIR)
      set(Clang_DIR ${LLVM_DIR})
    endif()
    if (NOT DEFINED Cling_DIR)
      set(Cling_DIR ${LLVM_DIR})
    endif()
    if (NOT DEFINED LLD_DIR)
      set(LLD_DIR ${LLVM_DIR})
    endif()
  endif()
  if (DEFINED LLD_DIR)
    if (NOT DEFINED LLVM_DIR)
      set(LLVM_DIR ${LLD_DIR})
    endif()
    if (NOT DEFINED Clang_DIR)
      set(Clang_DIR ${LLD_DIR})
    endif()
    if (NOT DEFINED Cling_DIR)
      set(Cling_DIR ${LLD_DIR})
    endif()
  endif()
  if (DEFINED Clang_DIR)
    if (NOT DEFINED LLVM_DIR)
      set(LLVM_DIR ${Clang_DIR})
    endif()
    if (NOT DEFINED LLD_DIR)
      set(LLD_DIR ${Clang_DIR})
    endif()
    if (NOT DEFINED Cling_DIR)
      set(Cling_DIR ${Clang_DIR})
    endif()
  endif()
  if (DEFINED Cling_DIR)
    if (NOT DEFINED LLVM_DIR)
      set(LLVM_DIR ${Cling_DIR})
    endif()
    if (NOT DEFINED Clang_DIR)
      set(Clang_DIR ${Cling_DIR})
    endif()
  endif()

include(GNUInstallDirs)
  ## Define supported version of clang and llvm

if (CPPINTEROP_USE_CLING)
  set(CLANG_MIN_SUPPORTED 18.0)
else()
  set(CLANG_MIN_SUPPORTED 16.0)
endif(CPPINTEROP_USE_CLING)
  set(CLANG_MAX_SUPPORTED "20.1.x")
  set(CLANG_VERSION_UPPER_BOUND 21.0.0)
if (CPPINTEROP_USE_CLING)
  set(LLD_MIN_SUPPORTED 18.0)
else()
  set(LLD_MIN_SUPPORTED 16.0)
endif(CPPINTEROP_USE_CLING)
  set(LLD_MAX_SUPPORTED "20.1.x")
  set(LLD_VERSION_UPPER_BOUND 21.0.0)
if (CPPINTEROP_USE_CLING)
  set(LLVM_MIN_SUPPORTED 18.0)
else()
  set(LLVM_MIN_SUPPORTED 16.0)
endif(CPPINTEROP_USE_CLING)
  set(LLVM_MAX_SUPPORTED "20.1.x")
  set(LLVM_VERSION_UPPER_BOUND 21.0.0)

  ## Set Cmake packages search order

  set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
  set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

  ## Search packages HINTS and PATHS

  if (DEFINED LLVM_DIR)
    set(llvm_search_hints PATHS ${LLVM_DIR} HINTS "${LLVM_DIR}/lib/cmake/llvm" "${LLVM_DIR}/cmake" "${LLVM_CONFIG_EXTRA_PATH_HINTS}")
    set(clang_search_hints PATHS ${LLVM_DIR} HINTS "${LLVM_DIR}/lib/cmake/clang" "${LLVM_DIR}/cmake")
    set(lld_search_hints PATHS ${LLVM_DIR} HINTS "${LLVM_DIR}/lib/cmake/lld" "${LLVM_DIR}/cmake")
  endif()
  if (DEFINED LLD_DIR)
    set(llvm_search_hints PATHS ${LLD_DIR} HINTS "${LLD_DIR}/lib/cmake/llvm" "${LLD_DIR}/cmake")
    set(lld_search_hints PATHS ${LLD_DIR} HINTS "${lld_search_hints}" "${LLD_DIR}/lib/cmake/lld" "${LLD_DIR}/cmake")
  endif()
  if (DEFINED Clang_DIR)
    set(llvm_search_hints PATHS ${Clang_DIR} HINTS "${Clang_DIR}/lib/cmake/llvm" "${Clang_DIR}/cmake")
    set(clang_search_hints PATHS ${Clang_DIR} HINTS "${clang_search_hints}" "${Clang_DIR}/lib/cmake/clang" "${Clang_DIR}/cmake" "${CLANG_CONFIG_EXTRA_PATH_HINTS}")
  endif()
  if (DEFINED Cling_DIR)
    set(cling_search_hints PATHS ${Cling_DIR} HINTS "${Cling_DIR}/lib/cmake/cling" "${Cling_DIR}/cmake" "${CLING_CONFIG_EXTRA_PATH_HINTS}")
  endif()

  ## Find supported LLVM

  if (CPPINTEROP_USE_CLING)
    message(STATUS "Mode CPPINTEROP_USE_CLING = ON")
    find_package(LLVM REQUIRED CONFIG ${llvm_search_hints} NO_DEFAULT_PATH)
    find_package(Clang REQUIRED CONFIG ${clang_search_hints} NO_DEFAULT_PATH)
    find_package(Cling REQUIRED CONFIG ${cling_search_hints} NO_DEFAULT_PATH)
  endif(CPPINTEROP_USE_CLING)

  if (LLVM_FOUND)
    if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
      unset(LLVM_FOUND)
      unset(LLVM_VERSION_MAJOR)
      unset(LLVM_VERSION_MINOR)
      unset(LLVM_VERSION_PATCH)
      unset(LLVM_PACKAGE_VERSION)
    else()
      if (NOT DEFINED LLVM_VERSION AND NOT DEFINED LLVM_DIR)
        set(LLVM_VERSION ${LLVM_PACKAGE_VERSION})
      endif()
    endif()
  endif()

  if (NOT LLVM_FOUND AND DEFINED LLVM_VERSION)
    if (LLVM_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
      set(LLVM_VERSION ${LLVM_VERSION_UPPER_BOUND})
    endif()
    if (LLVM_VERSION VERSION_LESS LLVM_MIN_SUPPORTED)
      set(LLVM_VERSION ${LLVM_MIN_SUPPORTED})
    endif()

    find_package(LLVM ${LLVM_VERSION} REQUIRED CONFIG ${llvm_search_hints} NO_DEFAULT_PATHS)
  endif()

  if (NOT LLVM_FOUND AND DEFINED LLVM_DIR)
    find_package(LLVM REQUIRED CONFIG ${llvm_search_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT LLVM_FOUND)
    find_package(LLVM REQUIRED CONFIG)
  endif()

  if (NOT LLVM_FOUND)
    message(FATAL_ERROR "Please set LLVM_DIR pointing to the LLVM build or installation folder")
  endif()

  if (LLVM_PACKAGE_VERSION VERSION_LESS LLVM_MIN_SUPPORTED OR LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL LLVM_VERSION_UPPER_BOUND)
    message(FATAL_ERROR "Found unsupported version: LLVM ${LLVM_PACKAGE_VERSION};\nPlease set LLVM_DIR pointing to the llvm version ${LLVM_MIN_SUPPORTED} to ${LLVM_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  ## Find supported LLD only while building for webassembly against emscripten

if(EMSCRIPTEN)
  if (DEFINED LLD_VERSION)
    if (LLD_VERSION VERSION_GREATER_EQUAL LLD_VERSION_UPPER_BOUND)
      set(LLD_VERSION ${LLD_VERSION_UPPER_BOUND})
    endif()
    if (LLD_VERSION VERSION_LESS LLD_MIN_SUPPORTED)
      set(LLD_VERSION ${LLD_MIN_SUPPORTED})
    endif()

    find_package(LLD ${LLD_VERSION} REQUIRED CONFIG ${lld_search_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT LLD_FOUND AND DEFINED LLD_DIR)
    find_package(LLD REQUIRED CONFIG ${lld_search_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT LLD_FOUND)
    find_package(LLD REQUIRED CONFIG)
  endif()

  if (NOT LLD_FOUND)
    message(FATAL_ERROR "Please set LLD_DIR pointing to the LLD build or installation folder")
  endif()

  set(LLD_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
  set(LLD_VERSION_MINOR ${LLVM_VERSION_MINOR})
  set(LLD_VERSION_PATCH ${LLVM_VERSION_PATCH})
  set(LLD_PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

  if (LLD_PACKAGE_VERSION VERSION_LESS LLD_MIN_SUPPORTED OR LLD_PACKAGE_VERSION VERSION_GREATER_EQUAL LLD_VERSION_UPPER_BOUND)
    message(FATAL_ERROR "Found unsupported version: LLD ${LLD_PACKAGE_VERSION};\nPlease set LLD_DIR pointing to the LLD version ${LLD_MIN_SUPPORTED} to ${LLD_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: LLD ${LLD_PACKAGE_VERSION}")
  message(STATUS "Using LLDConfig.cmake in: ${LLD_DIR}")
endif()

  ## Find supported Clang

  if (DEFINED CLANG_VERSION)
    if (CLANG_VERSION VERSION_GREATER_EQUAL CLANG_VERSION_UPPER_BOUND)
      set(CLANG_VERSION ${CLANG_VERSION_UPPER_BOUND})
    endif()
    if (CLANG_VERSION VERSION_LESS CLANG_MIN_SUPPORTED)
      set(CLANG_VERSION ${CLANG_MIN_SUPPORTED})
    endif()

    find_package(Clang ${CLANG_VERSION} REQUIRED CONFIG ${clang_search_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT Clang_FOUND AND DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG ${clang_search_hints} NO_DEFAULT_PATH)
  endif()

  if (NOT Clang_FOUND)
    find_package(Clang REQUIRED CONFIG)
  endif()

  if (NOT Clang_FOUND)
    message(FATAL_ERROR "Please set Clang_DIR pointing to the clang build or installation folder")
  endif()

  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
  set(CLANG_VERSION_PATCH ${LLVM_VERSION_PATCH})
  set(CLANG_PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

  if (CLANG_PACKAGE_VERSION VERSION_LESS CLANG_MIN_SUPPORTED OR CLANG_PACKAGE_VERSION VERSION_GREATER_EQUAL CLANG_VERSION_UPPER_BOUND)
    message(FATAL_ERROR "Found unsupported version: Clang ${CLANG_PACKAGE_VERSION};\nPlease set Clang_DIR pointing to the clang version ${CLANG_MIN_SUPPORTED} to ${CLANG_MAX_SUPPORTED} build or installation folder")
  endif()

  message(STATUS "Found supported version: Clang ${CLANG_PACKAGE_VERSION}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  ## Clang 16 require c++17 or later.
  if (NOT CMAKE_CXX_STANDARD)
    set (CMAKE_CXX_STANDARD 17)
  endif()
  if (CMAKE_CXX_STANDARD LESS 17)
    message(fatal "LLVM/CppInterOp requires c++17 or later")
  endif()

  ## Find supported Cling

  if (CPPINTEROP_USE_CLING)
    if (NOT Cling_FOUND AND DEFINED Cling_DIR)
      find_package(Cling REQUIRED CONFIG ${cling_extra_hints} NO_DEFAULT_PATH)
    endif()

    if (NOT Cling_FOUND)
      find_package(Cling REQUIRED CONFIG)
    endif()

    if (NOT Cling_FOUND)
      message(FATAL_ERROR "Please set Cling_DIR pointing to the cling build or installation folder")
    endif()

    message(STATUS "Found supported version: Cling ${CLING_PACKAGE_VERSION}")
    message(STATUS "Using ClingConfig.cmake in: ${Cling_DIR}")

  endif()

  #Replace \ with / in LLVM_DIR (attempt to fix path parsing issue Windows)
  string(REPLACE "\\" "/" LLVM_DIR "${LLVM_DIR}")

  # When in debug mode the llvm package thinks it is built with -frtti.
  # For consistency we should set it to the correct value.
  set(LLVM_CONFIG_HAS_RTTI NO CACHE BOOL "" FORCE)

  ## Init

  # In case this was a path to a build folder of llvm still try to find AddLLVM
  list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

  # Fix bug in some AddLLVM.cmake implementation (-rpath "" problem)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  set( CPPINTEROP_BUILT_STANDALONE 1 )
endif()

include(AddLLVM)
include(HandleLLVMOptions)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# In rare cases we might want to have clang installed in a different place
# than llvm and the header files should be found first (even though the
# LLVM_INCLUDE_DIRS) contain clang headers, too.
if(CPPINTEROP_USE_CLING)
  add_definitions(-DCPPINTEROP_USE_CLING)
  include_directories(SYSTEM ${CLING_INCLUDE_DIRS})
elseif(CPPINTEROP_USE_REPL)
  add_definitions(-DCPPINTEROP_USE_REPL)
else()
  message(FATAL_ERROR "We need either CPPINTEROP_USE_CLING or CPPINTEROP_USE_REPL")
endif()

include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# If the llvm sources are present add them with higher priority.
if (LLVM_BUILD_MAIN_SRC_DIR)
  # LLVM_INCLUDE_DIRS contains the include paths to both LLVM's source and
  # build directories. Since we cannot just include ClangConfig.cmake (see
  # fixme above) we have to do a little more work to get the right include
  # paths for clang.
  #
  # FIXME: We only support in-tree builds of clang, that is clang being built
  # in llvm_src/tools/clang.
  include_directories(SYSTEM ${LLVM_BUILD_MAIN_SRC_DIR}/tools/clang/include/)

  if (NOT LLVM_BUILD_BINARY_DIR)
    message(FATAL "LLVM_BUILD_* values should be available for the build tree")
  endif()

  include_directories(SYSTEM ${LLVM_BUILD_BINARY_DIR}/tools/clang/include/)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

## Code Coverage Configuration
add_library(coverage_config INTERFACE)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
  if(NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(FATAL_ERROR "CodeCov enabled on non-debug build!")
  endif()
  set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
  set(GCC_COVERAGE_LINK_FLAGS    "--coverage")
  set(CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHAREDLINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  if (APPLE OR EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
  endif ()
endif ()

# Fixes "C++ exception handler used, but unwind semantics are not enabled" warning Windows
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif ()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif ()

# FIXME: Use merge this with the content from the LLVMConfig and ClangConfig.
if (NOT CPPINTEROP_BUILT_STANDALONE)
include_directories(BEFORE SYSTEM
  ${CMAKE_CURRENT_BINARY_DIR}/../clang/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../clang/include
  )
endif()

include_directories(BEFORE SYSTEM
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

#Removes flag due to issue with Google test download when LLVM_ENABLE_WERROR=On
string(REPLACE "-Wcovered-switch-default" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

file(STRINGS "VERSION" CPPINTEROP_VERSION)
string(REPLACE "." ";" VERSION_LIST "${CPPINTEROP_VERSION}")
list(GET VERSION_LIST 0 CPPINTEROP_VERSION_MAJOR)
list(GET VERSION_LIST 1 CPPINTEROP_VERSION_MINOR)
list(GET VERSION_LIST 2 CPPINTEROP_VERSION_PATCH)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppInterOp/CppInterOpConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/CppInterOp/CppInterOpConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppInterOp/CppInterOpConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/CppInterOp/CppInterOpConfigVersion.cmake
  @ONLY)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/CppInterOp/
  DESTINATION lib/cmake/CppInterOp
  FILES_MATCHING
  PATTERN "*.cmake"
  )


install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY tools/
  DESTINATION include/CppInterOp/tools
  FILES_MATCHING
  PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "*.inc"
  )

add_definitions( -D_GNU_SOURCE )

# Add deps if we build together with clang.
if (TARGET intrinsics_gen)
  list(APPEND LLVM_COMMON_DEPENDS intrinsics_gen)
endif()
if (TARGET clang-headers)
  list(APPEND LLVM_COMMON_DEPENDS clang-headers)
endif()

# Generate docs for CppInterOp
option(CPPINTEROP_INCLUDE_DOCS "Generate build targets for the CppInterOp docs.")
option(CPPINTEROP_ENABLE_DOXYGEN "Use doxygen to generate CppInterOp internal API documentation.")
option(CPPINTEROP_ENABLE_SPHINX "Use sphinx to generate CppInterOp user documentation")


if(MSVC)

  set(MSVC_EXPORTLIST
    _Init_thread_header
    _Init_thread_footer
    ?nothrow@std@@3Unothrow_t@1@B
    ??_7type_info@@6B@
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # new/delete variants needed when linking to static msvc runtime (esp. Debug)
    set(MSVC_EXPORTLIST ${MSVC_EXPORTLIST}
      ??2@YAPEAX_K@Z
      ??3@YAXPEAX@Z
      ??_U@YAPEAX_K@Z
      ??_V@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
      ??_V@YAXPEAX_K@Z
      ??2@YAPEAX_KAEBUnothrow_t@std@@@Z
      ??_U@YAPEAX_KAEBUnothrow_t@std@@@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z
      ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z
    )
  else()
    set(MSVC_EXPORTLIST ${MSVC_EXPORTLIST}
      ??2@YAPAXI@Z
      ??3@YAXPAX@Z
      ??3@YAXPAXI@Z
      ??_U@YAPAXI@Z
      ??_V@YAXPAX@Z
      ??_V@YAXPAXI@Z
      ??2@YAPAXIABUnothrow_t@std@@@Z
      ??_U@YAPAXIABUnothrow_t@std@@@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@PBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@P6AAAV01@AAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@PBD@Z
      ?_Facet_Register@std@@YAXPAV_Facet_base@1@@Z
    )
  endif()

  if(MSVC_VERSION LESS 1914)
    set(MSVC_EXPORTLIST ${MSVC_EXPORTLIST} ??3@YAXPAX0@Z ??_V@YAXPAX0@Z)
  endif()

  if(MSVC_VERSION GREATER_EQUAL 1936 AND NOT (CMAKE_C_COMPILER MATCHES "aarch64|arm"))
    set(MSVC_EXPORTLIST ${MSVC_EXPORTLIST}
        __std_find_trivial_1
        __std_find_trivial_2
        __std_find_trivial_4
        __std_find_trivial_8
    )
  endif()

foreach(sym ${MSVC_EXPORTLIST})
  set(MSVC_EXPORTS "${MSVC_EXPORTS} /EXPORT:${sym}")
endforeach(sym ${MSVC_EXPORTLIST})

endif()

if (CPPINTEROP_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

add_subdirectory(lib)
if (CPPINTEROP_ENABLE_TESTING)
  add_subdirectory(unittests)
endif(CPPINTEROP_ENABLE_TESTING)
