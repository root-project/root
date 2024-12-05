# Do not depend on CMake scripts from the rest of the ROOT build
set(CMAKE_MODULE_PATH "")

#--- Check if we need to build llvm and clang ------------------------------------------------------
if (builtin_llvm AND NOT builtin_clang)
  message(FATAL_ERROR "Cannot enable builtin LLVM, but provide external Clang")
endif()

if (NOT builtin_clang)
  message(WARNING "Due to ROOT-specific patches you need a special version of clang. You cannot use vanilla clang.")
endif()

#--Set the LLVM version required for ROOT-----------------------------------------------------------
set(ROOT_LLVM_VERSION_REQUIRED_MAJOR 18)

#---Define the way we want to build and what of llvm/clang/cling------------------------------------
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_FFI OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(CLANG_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(CLANG_BUILD_TOOLS OFF CACHE BOOL "")
# It looks like that turning off CLANG_BUILD_TOOLS is not enough.
set(CLANG_TOOL_ARCMT_TEST_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_CHECK_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_FORMAT_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_FORMAT_VS_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_FUZZER_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_IMPORT_TEST_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_RENAME_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_C_ARCMT_TEST_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_C_INDEX_TEST_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_DIAGTOOL_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_LIBCLANG_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_SCAN_BUILD_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_SCAN_VIEW_BUILD OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_TOOL_LLVM_AR_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD OFF CACHE BOOL "")
set(LLVM_FORCE_USE_OLD_TOOLCHAIN ON CACHE BOOL "")

# Prevent Clang from finding CUDA just to determine the architecture for OpenMP
# offloading, which we are not interested in. If the CUDA package was already
# found before (because the user turned on cuda or tmva-gpu), this will not have
# an effect, which is fine.
# (Note that the option is very counter-intuitive: We turn *on* disabling it...)
# Except when we want to build cling with adaptiveCpp, in which case adaptiveCpp
# needs to find CUDA in order to build the backend.
if(NOT experimental_adaptivecpp)
  set(CMAKE_DISABLE_FIND_PACKAGE_CUDA ON)
endif()

# will be set again in case NOT builtin_llvm
set(LLVM_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm")
if (clad)
  set(CLING_BUILD_PLUGINS ON)
endif()

# We only use llvm/clang through TCling which is (with the help of core/meta) already taking a lock
# to serialize access to llvm.  We can later review how to make this finer grained by using llvm's own locking
# mechanism.
set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")

set(CMAKE_REQUIRED_QUIET 1)  # Make the configuration of LLVM quiet

if(ROOT_ARCHITECTURE MATCHES arm64)
  set(ROOT_CLING_TARGET "AArch64")
elseif(ROOT_ARCHITECTURE MATCHES arm)
  set(ROOT_CLING_TARGET "ARM")
elseif(ROOT_ARCHITECTURE MATCHES ppc64)
  set(ROOT_CLING_TARGET "PowerPC")
elseif(ROOT_ARCHITECTURE MATCHES s390)
  set(ROOT_CLING_TARGET "SystemZ")
elseif(ROOT_ARCHITECTURE MATCHES riscv64)
  set(ROOT_CLING_TARGET "RISCV")
elseif(ROOT_ARCHITECTURE MATCHES loong64)
  set(ROOT_CLING_TARGET "LoongArch")
elseif(ROOT_ARCHITECTURE MATCHES freebsd OR ROOT_ARCHITECTURE MATCHES linux OR ROOT_ARCHITECTURE MATCHES macosx)
  set(ROOT_CLING_TARGET "X86")
elseif(ROOT_ARCHITECTURE MATCHES win32 OR ROOT_ARCHITECTURE MATCHES win64)
  set(ROOT_CLING_TARGET "X86")
else()
  set(ROOT_CLING_TARGET "all")
endif()

if(NOT "${ROOT_CLING_TARGET}" STREQUAL "all")
  string(APPEND ROOT_CLING_TARGET ";NVPTX")
endif()

if(MSVC)
  # FIXME: since Visual Studio v16.4.0 the /O2 optimization flag make many (25%) of the tests failing
  # Try to re-enable /O2 after the upgrade of llvm & clang
  if (MSVC_VERSION GREATER_EQUAL 1924 AND MSVC_VERSION LESS 1929)
    string(REPLACE "-O2" "-O1 -Oi" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-O2" "-O1 -Oi" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  endif()
  # replace dashes in the -EH* and -GR* flags with slashes (/EH* /GR*)
  string(REPLACE " -EH" " /EH" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -GR" " /GR" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049,4206,4217,4221")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4049,4206,4217,4221")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /ignore:4049,4206,4217,4221")
endif()

set(LLVM_TARGETS_TO_BUILD ${ROOT_CLING_TARGET} CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")

if(clingtest)
  message("-- cling test suite enabled: llvm / clang symbols in libCling will be visible!")
  set(CLING_INCLUDE_TESTS ON CACHE BOOL "" FORCE)
  # The path to cling passed through environment variable only relevant when building
  # against external llvm. In that case, cling's testsuite cannot deduce the binary
  # of cling relatively to the llvm tools folder.
  if (NOT builtin_llvm)
    set(CLINGTEST_EXECUTABLE CLING=${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/${CMAKE_CFG_INTDIR}/bin/cling)
  endif()
  ROOT_ADD_TEST(clingtest-check-cling COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-cling
                                       ENVIRONMENT ${CLINGTEST_EXECUTABLE})
else()
  #---Build LLVM/Clang with symbol visibility=hidden--------------------------------------------------
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
endif()
if (NOT MSVC AND NOT APPLE)
  # Requires the linker to resolve the symbol internally and prevents
  # conflicts when linked with another software using also LLVM like in
  # the problem reported for Julia in
  # https://github.com/JuliaHEP/ROOT.jl/issues/17#issuecomment-882719292
  # Only needed for Linux: Mac uses linker namespaces and  Windows explicit export/import
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-Bsymbolic")
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS "-fno-semantic-interposition")
endif()
set(CMAKE_VISIBILITY_INLINES_HIDDEN "ON")

if(gcctoolchain)
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS --gcc-toolchain=${gcctoolchain})
endif()

# We will not fix llvm or clang.
string(REPLACE "-Werror " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
# Turn off coverage - we don't need this for llvm.
string(REPLACE "${GCC_COVERAGE_COMPILE_FLAGS}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


if(LLVM_SHARED_LINKER_FLAGS)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LLVM_SHARED_LINKER_FLAGS}")
endif()
if(LLVM_EXE_LINKER_FLAGS)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LLVM_EXE_LINKER_FLAGS}")
endif()

# Explicitly disable -Wmisleading-indentation for GCC: Some LLVM source files
# are too large.
set(cxx_flags_prev ${CMAKE_CXX_FLAGS})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation")
endif()

if(builtin_llvm)
  # Since debug builds of LLVM are quite large, we want to be able
  # to control the build types of ROOT and LLVM independently. The
  # logic below is to make that possible. LLVM is built in Release
  # mode unless a different build type is chosen via LLVM_BUILD_TYPE.

  if(NOT DEFINED LLVM_BUILD_TYPE)
    set(LLVM_BUILD_TYPE Release CACHE STRING "Build type used for LLVM")
  endif()

  message(STATUS "Building LLVM in '${LLVM_BUILD_TYPE}' mode.")

  if(NOT DEFINED LLVM_ENABLE_ASSERTIONS)
    if(CMAKE_BUILD_TYPE MATCHES "Debug"
     OR LLVM_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)")
      set(LLVM_ENABLE_ASSERTIONS TRUE)
      if(NOT DEFINED LLVM_ENABLE_DUMP)
        set(LLVM_ENABLE_DUMP TRUE)
      endif()
    else()
      set(LLVM_ENABLE_ASSERTIONS FALSE)
      set(LLVM_ENABLE_DUMP FALSE)
    endif()
  endif()

  # Regardless of LLVM_BUILD_TYPE and LLVM_ENABLE_ASSERTIONS, we never want to
  # "optimize llvm_unreachable() as undefined behavior" but trap if we hit this
  # in production builds.
  set(LLVM_UNREACHABLE_OPTIMIZE FALSE)

  set(_build_type ${CMAKE_BUILD_TYPE})

  # Multi-configuration generators ignore CMAKE_BUILD_TYPE, so
  # in that case we set the flags for all configurations to the
  # flags of the build type assigned to LLVM_BUILD_TYPE.

  if(MSVC OR XCODE)
    string(TOUPPER ${LLVM_BUILD_TYPE} LLVM_BUILD_TYPE)
    set(LLVM_C_FLAGS ${CMAKE_C_FLAGS_${LLVM_BUILD_TYPE}})
    set(LLVM_CXX_FLAGS ${CMAKE_CXX_FLAGS_${LLVM_BUILD_TYPE}})
    # On Windows, use the same compiler flags than ROOT and not
    # the other way around
    if(NOT MSVC)
      foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG)
        set(CMAKE_C_FLAGS_${CONFIG} ${LLVM_C_FLAGS})
        set(CMAKE_CXX_FLAGS_${CONFIG} ${LLVM_CXX_FLAGS})
      endforeach()
    endif()
  elseif(NOT LLVM_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE ${LLVM_BUILD_TYPE})
  endif()

  set(BUILD_SHARED_LIBS FALSE)

  #---Remove the inherited include_directories()
  set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")

  set(LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")

  if(experimental_adaptivecpp)
    # Register AdaptiveCpp as an external project
    set(LLVM_EXTERNAL_PROJECTS "AdaptiveCpp" CACHE STRING "")
    set(LLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR "${ADAPTIVE_CPP_SOURCE_DIR}" CACHE STRING "")
    set(LLVM_ADAPTIVECPP_LINK_INTO_TOOLS ON CACHE BOOL "") # To force-link the plugin into clang
  endif()

  # Always build LLVM with C++17. It is not necessary to compile with the same
  # C++ standard as the rest of ROOT and sometimes it doesn't even work.
  set(_cxx_standard ${CMAKE_CXX_STANDARD})
  set(CMAKE_CXX_STANDARD 17)

  #---Reduce log level to suppress STATUS messages from LLVM
  if(NOT DEFINED CMAKE_MESSAGE_LOG_LEVEL)
    set(CMAKE_MESSAGE_LOG_LEVEL "NOTICE")
    set(_unset_log_level TRUE)
  endif()

  #---Add the sub-directory excluding all the targets from all-----------------------------------------
  if(CMAKE_GENERATOR MATCHES "Xcode")
    add_subdirectory(llvm-project/llvm)
  else()
    add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)
  endif()

  if(DEFINED _unset_log_level)
    unset(CMAKE_MESSAGE_LOG_LEVEL)
  endif()

  set(CMAKE_CXX_STANDARD ${_cxx_standard})
  set(CMAKE_BUILD_TYPE ${_build_type})

  set(LLVM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/include
    ${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include
    CACHE STRING "LLVM include directories."
    )
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/cmake/modules")
  #---Set into parent scope LLVM_VERSION_MAJOR --------------------------------------------------------
  get_directory_property(_llvm_version_major DIRECTORY llvm-project/llvm DEFINITION LLVM_VERSION_MAJOR)
  set(LLVM_VERSION_MAJOR "${_llvm_version_major}" PARENT_SCOPE)
  #---Forward LLVM_TARGET_TRIPLE for check-cling-------------------------------------------------------
  get_directory_property(LLVM_TARGET_TRIPLE DIRECTORY llvm-project/llvm DEFINITION LLVM_TARGET_TRIPLE)
  #---Get back the potentially updated LLVM_TARGETS_TO_BUILD (expanding all/host/Native)---------------
  get_directory_property(LLVM_TARGETS_TO_BUILD DIRECTORY llvm-project/llvm DEFINITION LLVM_TARGETS_TO_BUILD)

  if(MSVC)
    if(CMAKE_GENERATOR MATCHES Ninja)
      set(LLVM_TABLEGEN_EXE "${LLVM_BINARY_DIR}/bin/llvm-tblgen.exe")
    else()
      set(LLVM_TABLEGEN_EXE "${LLVM_BINARY_DIR}/$<CONFIG>/bin/llvm-tblgen.exe")
    endif()
  else()
    set(LLVM_TABLEGEN_EXE "${LLVM_BINARY_DIR}/bin/llvm-tblgen")
  endif()
else()
  find_package(LLVM REQUIRED CONFIG)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_CMAKE_DIR}")

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})
  include_directories(${LLVM_INCLUDE_DIRS})

  # Make some LLVM variables globally available.
  set(LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS} CACHE PATH "Path to llvm/include")
  set(LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_LIBDIR_SUFFIX ${LLVM_LIBDIR_SUFFIX} CACHE STRING "Suffix of library directory name")

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  # We already FORCE-d the CACHE value to OFF, but LLVMConfig.cmake might have
  # set the variable to ON again...
  set(LLVM_ENABLE_WARNINGS OFF)

  # They are used as destination of target generators.
  # We try to keep these locations consistent with the builtin_llvm. This is important
  # for the LLVMRES target.
  # FIXME: In longer term, we do not really need this and may want to adjust LLVMRES.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/${CMAKE_CFG_INTDIR}/lib)

  if(WIN32 OR CYGWIN)
    # DLL platform -- put DLLs into bin.
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  endif()

  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)
  option(CLANG_ENABLE_BOOTSTRAP "Generate the clang bootstrap target" OFF)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
  include(VersionFromVCS)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
  if (${PACKAGE_VERSION} MATCHES "${ROOT_LLVM_VERSION_REQUIRED_MAJOR}\\.1(|\\.[0-9]+)")
    message(STATUS "Using LLVM external library - ${PACKAGE_VERSION}")
  else()
    message(FATAL_ERROR "LLVM version ${LLVM_PACKAGE_VERSION} different from ROOT supported, please try ${ROOT_LLVM_VERSION_REQUIRED_MAJOR}.1.x")
  endif()

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  if(LLVM_INCLUDE_TESTS OR clingtest)
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

    # Check prebuilt llvm/utils.
    if(EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()
    set(ROOT_LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/")
    if(EXISTS ${ROOT_LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      # Note: path not really used, except for checking if lit was found
      set(LLVM_LIT ${ROOT_LLVM_MAIN_SRC_DIR}/utils/lit/lit.py CACHE PATH "The location of the lit test runner.")
      if(NOT LLVM_UTILS_PROVIDED)
        add_subdirectory(${ROOT_LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
        add_subdirectory(${ROOT_LLVM_MAIN_SRC_DIR}/utils/count utils/count)
        add_subdirectory(${ROOT_LLVM_MAIN_SRC_DIR}/utils/not utils/not)
        set(LLVM_UTILS_PROVIDED ON)
        set(CLANG_TEST_DEPS FileCheck count not)
      endif()
      # We do not run llvm tests in ROOT.
      #set(UNITTEST_DIR ${ROOT_LLVM_MAIN_SRC_DIR}/utils/unittest)
      #if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
      #    AND NOT EXISTS ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
      #    AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
      #  add_subdirectory(${UNITTEST_DIR} utils/unittest)
      #endif()
    else()
      # Seek installed Lit.
      find_program(LLVM_LIT
                   NAMES llvm-lit lit.py lit
                   PATHS "${ROOT_LLVM_MAIN_SRC_DIR}/utils/lit"
                   DOC "Path to lit.py")
    endif()

    if(LLVM_LIT)
      # Define the default arguments to use with 'lit', and an option for the user
      # to override.
      set(LIT_ARGS_DEFAULT "-sv")
      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
      set(LIT_COMMAND "${LLVM_LIT}" CACHE STRING "Path to the LLVM LIT.")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if( WIN32 AND NOT CYGWIN )
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()
    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()
  endif()

  # We checked above that LLVM_VERSION is what we require in ROOT_LLVM_VERSION_REQUIRED_MAJOR.
  # To simplify code, just forward that requirement to the rest of ROOT, for example to
  # construct the resource directory in core/clingutils.
  set(LLVM_VERSION_MAJOR ${ROOT_LLVM_VERSION_REQUIRED_MAJOR} PARENT_SCOPE)

  # We are in the case of NOT builtin_llvm
  if (builtin_clang)
    # remove clang-cpp from CLANG_LINKS_TO_CREATE to avoid clashes with
    # install-clang-cpp target defined by LLVM's cmake module
    set(CLANG_LINKS_TO_CREATE clang++ clang-cl)

    # Disable linking against shared LLVM
    set(LLVM_LINK_LLVM_DYLIB OFF)

    # Always build LLVM with C++17. It is not necessary to compile with the same
    # C++ standard as the rest of ROOT and sometimes it doesn't even work.
    set(_cxx_standard ${CMAKE_CXX_STANDARD})
    set(CMAKE_CXX_STANDARD 17)

    add_subdirectory(llvm-project/clang EXCLUDE_FROM_ALL)

    set(CMAKE_CXX_STANDARD ${_cxx_standard})
  endif(builtin_clang)

  set( CLANG_BUILT_STANDALONE 1 )
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
endif(builtin_llvm)

if (builtin_clang)
  if (builtin_llvm)
    set(CLANG_BINARY_DIR "${LLVM_BINARY_DIR}/tools/clang/")
  else()
    set(CLANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/clang/")
  endif()
  set(CLANG_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/clang/include
    ${CLANG_BINARY_DIR}/include
    CACHE STRING "Clang include directories.")
  set(CLANG_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/clang/")
else()
  find_package(Clang REQUIRED CONFIG)
  message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION} in ${CLANG_CMAKE_DIR}")

  # Forward CLANG_INSTALL_PREFIX so it can be used by core/clingutils/CMakeLists.txt
  set(CLANG_INSTALL_PREFIX ${CLANG_INSTALL_PREFIX} PARENT_SCOPE)
endif()

# Reset the compiler flags after compiling LLVM and Clang
set(CMAKE_CXX_FLAGS ${cxx_flags_prev})
unset(cxx_flags_prev)

add_custom_target(CLING)
set(CLING_LIBRARIES clingInterpreter;clingMetaProcessor;clingUtils CACHE STRING "")
if (builtin_cling)

  # Extract the compilation flags from LLVM and make them public to the
  # rest of ROOT so that we can compile against LLVM with matching flags.

  # LLVM doesn't really give us a API to get this with an in-source build
  # so we just use the normal way of doing this and read the llvm directory
  # compilation properties.
  if (builtin_llvm)
    get_directory_property(LLVM_DEFS DIRECTORY llvm-project/llvm COMPILE_DEFINITIONS)
    # Turns DEFINE1;DEFINE2 to -DDEFINE1 -DDEFINE2
    string (REPLACE ";" " -D" LLVM_DEFS ";${LLVM_DEFS}")
  endif()

  # FIXME: Reduce the usage of CLING_CXXFLAGS by adding a cmake routine in
  # RootMacros.cmake for all cling-dependent libraries
  if(MSVC)
    set(CLING_CXXFLAGS " ${LLVM_DEFS} -DNOMINMAX -D_XKEYCHECK_H")
  else()
    # FIXME: Work hard to remove -Wno-shadow and -Wno-unused-parameter
    set(CLING_CXXFLAGS " ${LLVM_DEFS} -fno-strict-aliasing -Wwrite-strings -Wno-shadow -Wno-unused-parameter")
  endif()

  # Set the flags in the parent scope for the rest of the cling-based libraries in ROOT.
  set(CLING_CXXFLAGS ${CLING_CXXFLAGS} PARENT_SCOPE)

  string(APPEND CMAKE_CXX_FLAGS ${CLING_CXXFLAGS})
  if (LLVM_ENABLE_PIC AND NOT MSVC)
    # FIXME: LLVM_ENABLE_PIC is ON by default, however not propagated to cling.
    # FIXME: fPIC is required by the linker for libCling. Can we do that using
    # cmake: set_property(TARGET clingInterpreter PROPERTY POSITION_INDEPENDENT_CODE ON)?
    string(APPEND CMAKE_CXX_FLAGS " -fPIC")
    string(APPEND CMAKE_C_FLAGS " -fPIC")
  endif(LLVM_ENABLE_PIC AND NOT MSVC)
  # Avoid cling being installed under ROOTSYS/include.
  set(CLING_ROOT_BUILD ON)
  add_subdirectory(cling EXCLUDE_FROM_ALL)
  add_dependencies(CLING ${CLING_LIBRARIES})

  set(CLING_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/cling/include CACHE STRING "")



  #---These are the libraries that we link ROOT with CLING---------------------------
  if (builtin_llvm)
    add_dependencies(CLING intrinsics_gen)
  endif()
  if (builtin_clang)
    add_dependencies(CLING clang-headers clang-resource-headers)
    # Windows (and some other LLVM setups) do not have that target available.
    if (TARGET clang-cmake-exports)
      add_dependencies(CLING clang-cmake-exports)
    endif()
  endif()
else()
  set(Cling_DIR ${LLVM_BINARY_DIR}/lib/cmake/cling/)
  find_package(Cling REQUIRED CONFIG PATHS ${Cling_DIR} "${Cling_DIR}/lib/cmake/cling" "${Cling_DIR}/cmake" NO_DEFAULT_PATH)
  find_package(Clang REQUIRED CONFIG PATHS ${Cling_DIR} "${Cling_DIR}/lib/cmake/clang" "${Cling_DIR}/cmake" NO_DEFAULT_PATH)

  # We need to consider not just the direct link dependencies, but also the
  # transitive link dependencies. Do this by starting with the set of direct
  # dependencies, then the dependencies of those dependencies, and so on.
  set(new_libs "clingMetaProcessor")
  set(link_libs ${new_libs})
  while(NOT "${new_libs}" STREQUAL "")
    foreach(lib ${new_libs})
      if(TARGET ${lib})
        get_target_property(lib_type ${lib} TYPE)
        if("${lib_type}" STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${lib})
        else()
          list(APPEND other_libs ${lib})
        endif()
        get_target_property(transitive_libs ${lib} INTERFACE_LINK_LIBRARIES)
        if (NOT transitive_libs)
          continue()
        endif()
        foreach(transitive_lib ${transitive_libs})
          list(FIND link_libs ${transitive_lib} idx)
          #if(TARGET ${transitive_lib} AND idx EQUAL -1)
          if(idx EQUAL -1)
            list(APPEND newer_libs ${transitive_lib})
            list(APPEND link_libs ${transitive_lib})
          endif()
        endforeach(transitive_lib)
      endif()
    endforeach(lib)
    set(new_libs ${newer_libs})
    set(newer_libs "")
  endwhile()
  set(CLING_DEPEND_LIBS ${link_libs} CACHE STRING "")
endif(builtin_cling)



#---Avoid to dependencies to system (e.g. atomic) libraries without modifying the LLVM code----------
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  get_target_property(__libs LLVMSupport INTERFACE_LINK_LIBRARIES)
  list(REMOVE_ITEM __libs atomic)
  set_target_properties(LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES "${__libs}")
endif()

#---Mark the LLVM/CLANG variables as advanced--------------------------------------------------------
get_cmake_property(variables CACHE_VARIABLES)
foreach(var ${variables})
  if(var MATCHES "^(CLANG|LLVM|CLING)_")
    mark_as_advanced(FORCE ${var})
  endif()
endforeach()
mark_as_advanced(FORCE BUG_REPORT_URL BUILD_CLANG_FORMAT_VS_PLUGIN BUILD_SHARED_LIBS BUILD_TESTING
                       C_INCLUDE_DIRS DEFAULT_SYSROOT FFI_INCLUDE_DIR FFI_LIBRARY_DIR
                       GCC_INSTALL_PREFIX LIBCLANG_BUILD_STATIC TOOL_INFO_PLIST)
mark_as_advanced(CLEAR LLVM_ENABLE_ASSERTIONS LLVM_BUILD_TYPE)

##################### LIBINTEROP ###########################

#---Set InterOp include directories, for any ROOT component that requires the headers--------------------------------------------------------
set(CPPINTEROP_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/CppInterOp/include
  CACHE STRING "CppInterOp include directories.")

#---Set InterOp to build on Cling, this can be toggled to use Clang-REPL-------------------------------------------------------
set(CPPINTEROP_USE_CLING ON CACHE BOOL "" FORCE)
set(CPPINTEROP_USE_REPL OFF CACHE BOOL "" FORCE)

#---Disable testing on InterOp if ROOT's testing is OFF (InterOp tests are enabled by default) -------------------------------------------------------
ROOT_CHECK_CONNECTION("CPPINTEROP_ENABLE_TESTING=OFF")
if(testing AND NOT NO_CONNECTION)
  set(CPPINTEROP_ENABLE_TESTING ON CACHE BOOL "" FORCE)
else()
  if(NO_CONNECTION)
    message(STATUS "No internet connection, disabling the `CppInterOp testing infrastructure` option")
  endif()
  set(CPPINTEROP_ENABLE_TESTING OFF CACHE BOOL "")
endif()

add_subdirectory(CppInterOp)
if(TARGET CppInterOpUnitTests)
  set_target_properties(CppInterOpUnitTests PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()
