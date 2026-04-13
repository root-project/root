set(gsl_version 2.8)
message(STATUS "Downloading and building GSL version ${gsl_version}")
set(GSL_LIBRARIES ${GSL_LIBRARIES})
foreach(l gsl gslcblas)
  list(APPEND GSL_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${l}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()
set(GSL_CBLAS_LIBRARY ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gslcblas${CMAKE_STATIC_LIBRARY_SUFFIX})
if(CMAKE_OSX_SYSROOT)
  set(_gsl_cppflags "-isysroot ${CMAKE_OSX_SYSROOT}")
  set(_gsl_ldflags  "-isysroot ${CMAKE_OSX_SYSROOT}")
endif()
ExternalProject_Add(
  GSL
  # http://mirror.switch.ch/ftp/mirror/gnu/gsl/gsl-${gsl_version}.tar.gz
  URL ${lcgpackages}/gsl-${gsl_version}.tar.gz
  URL_HASH SHA256=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190
  SOURCE_DIR GSL-src # prevent "<gsl/...>" vs GSL/ macOS warning
  INSTALL_DIR ${CMAKE_BINARY_DIR}
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR>
                    --libdir=<INSTALL_DIR>/lib
                    --enable-shared=no --with-pic
                    CC=${CMAKE_C_COMPILER}
                    CFLAGS=${CMAKE_C_FLAGS}
                    CPPFLAGS=${_gsl_cppflags}
                    LDFLAGS=${_gsl_ldflags}
  LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 LOG_OUTPUT_ON_FAILURE 1
  BUILD_BYPRODUCTS ${GSL_LIBRARIES}
  TIMEOUT 600
)
set(GSL_TARGET GSL PARENT_SCOPE)
# FIXME: one need to find better way to extract path with GSL include files
set(GSL_INCLUDE_DIR ${CMAKE_BINARY_DIR}/GSL-prefix/src/GSL-build PARENT_SCOPE)
set(GSL_FOUND ON PARENT_SCOPE)
