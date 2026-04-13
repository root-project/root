set(ROOT_GSL_VERSION 2.8)
message(STATUS "Downloading and building GSL version ${ROOT_GSL_VERSION}")
set(ROOT_GSL_PREFIX ${CMAKE_BINARY_DIR}/builtins/GSL-prefix)
set(ROOT_GSL_LIBRARY ${ROOT_GSL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gsl${CMAKE_STATIC_LIBRARY_SUFFIX})
set(ROOT_GSL_CBLAS_LIBRARY ${ROOT_GSL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gslcblas${CMAKE_STATIC_LIBRARY_SUFFIX})
set(ROOT_GSL_LIBRARIES ${ROOT_GSL_LIBRARY} ${ROOT_GSL_CBLAS_LIBRARY})
if(CMAKE_OSX_SYSROOT)
  set(_gsl_cppflags "-isysroot ${CMAKE_OSX_SYSROOT}")
  set(_gsl_ldflags  "-isysroot ${CMAKE_OSX_SYSROOT}")
endif()
ExternalProject_Add(
  BUILTIN_GSL
  PREFIX ${ROOT_GSL_PREFIX}
  # http://mirror.switch.ch/ftp/mirror/gnu/gsl/gsl-${ROOT_GSL_VERSION}.tar.gz
  URL ${lcgpackages}/gsl-${ROOT_GSL_VERSION}.tar.gz
  URL_HASH SHA256=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190
  SOURCE_DIR GSL-src # prevent "<gsl/...>" vs GSL/ macOS warning
  INSTALL_DIR ${ROOT_GSL_PREFIX}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR>
                    --libdir=<INSTALL_DIR>/lib
                    --enable-shared=no --with-pic
                    CC=${CMAKE_C_COMPILER}
                    CFLAGS=${CMAKE_C_FLAGS}
                    CPPFLAGS=${_gsl_cppflags}
                    LDFLAGS=${_gsl_ldflags}
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${ROOT_GSL_LIBRARIES}
  TIMEOUT 600
)

# FIXME: one need to find better way to extract path with GSL include files
set(ROOT_GSL_INCLUDE_DIR ${ROOT_GSL_PREFIX}/src/BUILTIN_GSL-build)

file(MAKE_DIRECTORY ${ROOT_GSL_INCLUDE_DIR})
add_library(GSL::gsl IMPORTED STATIC GLOBAL)
add_dependencies(GSL::gsl BUILTIN_GSL)
set_target_properties(GSL::gsl PROPERTIES
    IMPORTED_LOCATION ${ROOT_GSL_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_GSL_INCLUDE_DIR})
add_library(GSL::gslcblas IMPORTED STATIC GLOBAL)
add_dependencies(GSL::gslcblas BUILTIN_GSL)
set_target_properties(GSL::gslcblas PROPERTIES
    IMPORTED_LOCATION ${ROOT_GSL_CBLAS_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ROOT_GSL_INCLUDE_DIR})

set(GSL_INCLUDE_DIRS ${ROOT_GSL_INCLUDE_DIR})
set(GSL_FOUND ON PARENT_SCOPE)
set(GSL_VERSION ${ROOT_GSL_VERSION} PARENT_SCOPE)
set(GSL_LIBRARIES ${ROOT_GSL_LIBRARIES} PARENT_SCOPE)
