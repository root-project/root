set(tbb_url ${lcgpackages}/oneTBB-2021.9.0.tar.gz)
set(tbb_sha256 1ce48f34dada7837f510735ff1172f6e2c261b09460e3bf773b49791d247d24e)

if(MSVC)
  if(CMAKE_GENERATOR MATCHES Ninja)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
      set(tbbsuffix "_debug")
    endif()
  else()
    set(tbb_build Release)
    if(winrtdebug)
      set(tbb_build Debug)
      set(tbbsuffix "_debug")
    endif()
  endif()
  set(TBB_LIBRARIES ${CMAKE_BINARY_DIR}/lib/tbb12${tbbsuffix}.lib)
  set(TBB_CXXFLAGS "-D__TBB_NO_IMPLICIT_LINKAGE=1")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/ DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libraries FILES_MATCHING PATTERN "tbb*.dll")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "tbb*.lib")
else()
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(tbbsuffix "_debug")
  endif()
  set(TBB_LIBRARIES ${CMAKE_BINARY_DIR}/lib/libtbb${tbbsuffix}${CMAKE_SHARED_LIBRARY_SUFFIX})
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libraries FILES_MATCHING PATTERN "libtbb*")
endif()
if(tbb_build)
  set(TBB_EXTRA_BUILD_ARGS --config ${tbb_build})
endif()

ExternalProject_Add(
  TBB
  URL ${tbb_url}
  URL_HASH SHA256=${tbb_sha256}
  INSTALL_DIR ${CMAKE_BINARY_DIR}
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_POLICY_VERSION_MINIMUM=3.5
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${ROOT_EXTERNAL_CXX_FLAGS}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_BINARY_DIR}/include
             -DCMAKE_INSTALL_LIBDIR=${CMAKE_BINARY_DIR}/lib
             -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
             -DTBBMALLOC_BUILD=OFF
             -DTBBMALLOC_PROXY_BUILD=OFF
             -DTBB_ENABLE_IPO=OFF
             -DTBB_STRICT=OFF
             -DTBB_TEST=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${TBB_EXTRA_BUILD_ARGS}
  INSTALL_COMMAND ${CMAKE_COMMAND}  --install . ${TBB_EXTRA_BUILD_ARGS}
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${TBB_LIBRARIES}
  TIMEOUT 600
)

ExternalProject_Add_Step(
   TBB tbb2externals
   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include/tbb ${CMAKE_BINARY_DIR}/ginclude/tbb
   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/include/oneapi ${CMAKE_BINARY_DIR}/ginclude/oneapi
   DEPENDEES install
)
set(TBB_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/ginclude PARENT_SCOPE)
set(TBB_CXXFLAGS "-DTBB_SUPPRESS_DEPRECATED_MESSAGES=1" PARENT_SCOPE)
# The following line is needed to generate the proper dependency with: BUILTINS TBB (in Imt)
# and generated with this syntax: add_dependencies(${library} ${${arg1}_TARGET})
set(TBB_TARGET TBB PARENT_SCOPE)