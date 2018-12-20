include(ExternalProject)

set(PCRE_VERSION "8.42")
set(PCRE_URL "${CMAKE_CURRENT_SOURCE_DIR}/pcre-${PCRE_VERSION}.tar.gz")
set(PCRE_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}pcre${CMAKE_STATIC_LIBRARY_SUFFIX})
set(PCRE_LIBRARY ${CMAKE_BINARY_DIR}/lib/${PCRE_LIBNAME})

if(WIN32)
  if (winrtdebug)
    set(pcre_lib pcred.lib)
    set(pcre_build_type Debug)
  else()
    set(pcre_lib pcre.lib)
    set(pcre_build_type Release)
  endif()
  ExternalProject_Add(PCRE
    URL ${PCRE_URL}
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${pcre_build_type}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${pcre_build_type}/${pcre_lib} ${PCRE_LIBRARY}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcre.h  <INSTALL_DIR>/include
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcre_scanner.h  <INSTALL_DIR>/include
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcre_stringpiece.h  <INSTALL_DIR>/include
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcrecpp.h  <INSTALL_DIR>/include
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcrecpparg.h  <INSTALL_DIR>/include
            COMMAND ${CMAKE_COMMAND} -E copy_if_different pcreposix.h  <INSTALL_DIR>/include
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${PCRE_LIBRARY})
else()
  if(CMAKE_OSX_SYSROOT)
    set(_pcre_cflags "-isysroot ${CMAKE_OSX_SYSROOT}")
  endif()
  ExternalProject_Add(PCRE
    URL ${PCRE_URL}
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CONFIGURE_COMMAND ./configure --prefix <INSTALL_DIR> --with-pic --disable-shared
                      CC=${CMAKE_C_COMPILER} CFLAGS=${_pcre_cflags}
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${PCRE_LIBRARY})
endif()

unset(PCRE_FOUND CACHE)
unset(PCRE_FOUND PARENT_SCOPE)
set(PCRE_FOUND TRUE CACHE BOOL "" FORCE)

set(PCRE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include CACHE INTERNAL "" FORCE)
set(PCRE_LIBRARY ${PCRE_LIBRARY} CACHE INTERNAL "" FORCE)
set(PCRE_LIBRARIES ${PCRE_LIBRARY} CACHE INTERNAL "" FORCE)
set(PCRE_TARGET PCRE)

set_property(GLOBAL APPEND PROPERTY ROOT_BUILTIN_TARGETS PCRE)
