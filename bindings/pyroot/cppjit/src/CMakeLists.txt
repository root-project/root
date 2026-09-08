# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# ROOT build of the vendored cppjit tree. Upstream's scikit-build-core
# CMake (../CMakeLists.txt fetches CppInterOp as an external project) is
# vendored but unreferenced: CppInterOp comes from interpreter/CppInterOp,
# compiled into libCling, with its headers staged under etc/cppinterop.

# --- interop layer (the cppyy-backend equivalent) ----------------------------

add_library(cppjit_backend STATIC
    interop/interop_wrapper.cxx
    interop/cppinterop_dispatch.cxx)

target_include_directories(cppjit_backend PRIVATE
    ${CMAKE_BINARY_DIR}/etc/cppinterop
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpyrt
    ${CMAKE_CURRENT_SOURCE_DIR}/interop
)

target_compile_definitions(cppjit_backend PRIVATE
  # cppinterop_paths() coordinates: the include dir is joined onto the
  # install prefix, so this resolves to ${CMAKE_BINARY_DIR}/etc/cppinterop.
  # The dispatch API is loaded from libCling instead of the standalone
  # library; since that library never exists next to libcpyrt, the
  # relocation probe in cppinterop_paths() always falls back to this
  # prefix. FIXME: provide the staged header location robustly for the
  # install tree as well.
  CPPINTEROP_INSTALL_PREFIX="${CMAKE_BINARY_DIR}"
  CPPINTEROP_INCLUDE_DIR="etc/cppinterop"
  CPPINTEROP_LIBRARY="libclangCppInterOp${CMAKE_SHARED_LIBRARY_SUFFIX}"
  CPPJIT_CLANG_MAJOR="${LLVM_VERSION_MAJOR}"
  CPPJIT_CLANG_INCLUDE_DIR=""
  # The CppInterOp JitCall debug assertions (AreArgumentsValid,
  # ReportInvokeStart) use Clang AST internals and cannot be resolved
  # through the dispatch mechanism. Disable them in this translation unit.
  NDEBUG
  # libcpyrt owns the dispatch table (cppinterop_dispatch.cxx); consumers in
  # other DLLs (ROOTPythonizations, tpython) import it. On Windows the data
  # symbols need explicit dllexport - WINDOWS_EXPORT_ALL_SYMBOLS covers
  # functions only.
  CPPINTEROP_DISPATCH_EXPORTS
)

target_include_directories(cppjit_backend SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})

target_link_libraries(cppjit_backend Core)
# Ensure CppInterOpEtc responsible for staging the tablegen-generated .inc
# files is built before cppjit_backend.
add_dependencies(cppjit_backend CppInterOpEtc)

if(NOT MSVC)
  target_compile_options(cppjit_backend PRIVATE -fPIC)
endif()

# --- cpyrt (the CPyCppyy equivalent) ------------------------------------------

set(headers
    cpyrt/API.h
    cpyrt/Reflex.h
    cpyrt/CommonDefs.h
    cpyrt/PyException.h
    cpyrt/DispatchPtr.h
)

set(sources
    cpyrt/API.cxx
    cpyrt/CPPClassMethod.cxx
    cpyrt/CPPConstructor.cxx
    cpyrt/CPPDataMember.cxx
    cpyrt/CPPEnum.cxx
    cpyrt/CPPExcInstance.cxx
    cpyrt/CPPFunction.cxx
    cpyrt/CPPGetSetItem.cxx
    cpyrt/CPPInstance.cxx
    cpyrt/CPPMethod.cxx
    cpyrt/CPPOperator.cxx
    cpyrt/CPPOverload.cxx
    cpyrt/CPPScope.cxx
    cpyrt/cpyrtModule.cxx
    cpyrt/CallContext.cxx
    cpyrt/Converters.cxx
    cpyrt/CustomPyTypes.cxx
    cpyrt/DispatchPtr.cxx
    cpyrt/Dispatcher.cxx
    cpyrt/Executors.cxx
    cpyrt/LowLevelViews.cxx
    cpyrt/MemoryRegulator.cxx
    cpyrt/ProxyWrappers.cxx
    cpyrt/PyException.cxx
    cpyrt/PyStrings.cxx
    cpyrt/Pythonize.cxx
    cpyrt/TemplateProxy.cxx
    cpyrt/TupleOfInstances.cxx
    cpyrt/TypeManip.cxx
    cpyrt/Utility.cxx
)

add_library(cpyrt SHARED ${headers} ${sources})
# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(cpyrt PROPERTIES ${ROOT_LIBRARY_PROPERTIES})
if(MSVC)
  target_link_libraries(cpyrt PRIVATE cppjit_backend)
  target_link_libraries(cpyrt PUBLIC Python3::Python)
  set_target_properties(cpyrt PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  target_link_options(cpyrt PRIVATE "-ignore:4197")
elseif(APPLE)
  target_link_libraries(cpyrt PRIVATE -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup cppjit_backend)
else()
  target_link_libraries(cpyrt PRIVATE -Wl,--unresolved-symbols=ignore-all cppjit_backend)
endif()

if(NOT MSVC)
  target_compile_options(cpyrt PRIVATE -Wno-strict-aliasing)
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT MSVC)
  target_compile_options(cpyrt PRIVATE
    -Wno-unused-but-set-parameter)
endif()

add_library(cppjit SHARED cpyrt/cpyrtPyModule.cxx)

# Set the suffix to '.so' and the prefix to 'lib'
set_target_properties(cppjit PROPERTIES ${ROOT_LIBRARY_PROPERTIES_NO_VERSION}
  LIBRARY_OUTPUT_DIRECTORY ${localruntimedir}/cppjit)
if(MSVC)
  target_link_libraries(cppjit PRIVATE cpyrt)
  set_target_properties(cppjit PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set_target_properties(cppjit PROPERTIES PREFIX "lib")
  set_target_properties(cppjit PROPERTIES SUFFIX ".pyd")
  # On Windows the DLL of a SHARED target follows the RUNTIME output
  # directory; import cppjit.libcppjit needs it inside the package dir.
  set_target_properties(cppjit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${localruntimedir}/cppjit
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${localruntimedir}/cppjit
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${localruntimedir}/cppjit
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${localruntimedir}/cppjit)
elseif(APPLE)
  target_link_libraries(cppjit PRIVATE -Wl,-bind_at_load -Wl,-w -Wl,-undefined -Wl,dynamic_lookup cpyrt)
else()
  target_link_libraries(cppjit PRIVATE -Wl,--unresolved-symbols=ignore-all cpyrt)
endif()

# Avoid warnings due to invalid function casts from C++ functions in cpyrt
# to CPython API function typedefs (e.g. PyCFunction). This is a common pattern
# in CPython extension implementations, explicitly encouraged by the official
# CPython docs for C/C++ extensions. see
# https://docs.python.org/3/extending/extending.html#keyword-parameters-for-extension-functions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(cpyrt PRIVATE -Wno-cast-function-type)
endif()

# CPPINTEROP_DISPATCH_EXPORTS: cpyrt TUs live in the same DLL as the dispatch
# table absorbed from cppjit_backend, so they mark it dllexport too.
target_compile_definitions(cpyrt PRIVATE NO_CPPJIT_LEGACY_NAMESPACE
                                         CPPINTEROP_DISPATCH_EXPORTS)

target_include_directories(cpyrt SYSTEM PUBLIC ${Python3_INCLUDE_DIRS})

# The source dir resolves the public "cpyrt/*.h" spellings against the
# flattened cpyrt directory; the interop dir provides cppjit_interop.h,
# which uses CppInterOp types directly, so consumers also need the staged
# CppInterOp headers.
target_include_directories(cpyrt
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpyrt>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interop>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/etc/cppinterop>
)
add_dependencies(cpyrt CppInterOpEtc)

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cpyrt)
set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppjit)

if(NOT MSVC)
  # Make sure that relative RUNPATH to main ROOT libraries is always correct.
  ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(cppjit ${CMAKE_INSTALL_PYTHONDIR}/cppjit)
  ROOT_APPEND_LIBDIR_TO_INSTALL_RPATH(cpyrt ${CMAKE_INSTALL_LIBDIR})
  set(CPPJIT_INSTALL_DIR ${CMAKE_INSTALL_PYTHONDIR}/cppjit)
else()
  set(CPPJIT_INSTALL_DIR ${CMAKE_INSTALL_PYTHONDIR})
endif()

# Install library
install(TARGETS cpyrt EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries
                            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries
                            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries)

install(TARGETS cppjit EXPORT ${CMAKE_PROJECT_NAME}Exports
                            RUNTIME DESTINATION ${CPPJIT_INSTALL_DIR} COMPONENT libraries # Windows
                            LIBRARY DESTINATION ${CPPJIT_INSTALL_DIR} COMPONENT libraries
                            ARCHIVE DESTINATION ${CPPJIT_INSTALL_DIR} COMPONENT libraries)

file(COPY ${headers} DESTINATION ${CMAKE_BINARY_DIR}/include/cpyrt)

install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpyrt
        COMPONENT headers)
