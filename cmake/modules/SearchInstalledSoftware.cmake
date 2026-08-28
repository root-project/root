# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#----------------------------------------------------------------------------
# macro ROOT_CHECK_CONNECTION(option)
# Try to download a file to check internet connection.
# If fail-on-missing=ON is set, a failed connection check will cause a fatal
# configuration error.
# Input variables:
#    option:
#        A hint to the user on which option to set to avoid the part of the
#        configuration that requested the connection check.
# Output variables:
#    NO_CONNECTION:
#        This variable is set based on the result of the connection check:
#          - FALSE: An active internet connection was found.
#          - TRUE: No internet connection was found or the download failed.
# Note: if the value of NO_CONNECTION is already FALSE, when calling the
#       macro, the connection check will not run again.
#----------------------------------------------------------------------------
macro(ROOT_CHECK_CONNECTION option)
  # Do something only if connection check is not already done
  if(NOT DEFINED NO_CONNECTION)
    if(NOT check_connection)
      # If the connection check is disabled, just assume there is internet
      # connection
      set(NO_CONNECTION FALSE)
    else()
      message(STATUS "Checking internet connectivity")
      file(DOWNLOAD https://root.cern/files/cmake_connectivity_test.txt ${CMAKE_CURRENT_BINARY_DIR}/cmake_connectivity_test.txt
        TIMEOUT 10 STATUS DOWNLOAD_STATUS
      )
      # Get the status code from the download status
      list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
      # Check if download was successful.
      if(${STATUS_CODE} EQUAL 0)
        # Success
        message(STATUS "Checking internet connectivity - found")
        # Now let's delete the file
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/cmake_connectivity_test.txt)
        set(NO_CONNECTION FALSE)
      else()
        # Error
        if(fail-on-missing)
          message(FATAL_ERROR "No internet connection. Please check your connection, set '-D${option}' or disable 'fail-on-missing' to automatically disable options requiring internet access. You can also bypass the connection check with -Dcheck_connection=OFF.")
        endif()
        message(STATUS "Checking internet connectivity - failed: will not automatically download external dependencies. You can bypass the connection check with -Dcheck_connection=OFF.")
        set(NO_CONNECTION TRUE)
      endif()
    endif()
  endif()
endmacro()

#----------------------------------------------------------------------------
# macro ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION(option_name)
# Check internet connection. If no connection, either disable the option or
# stop the configuration with a FATAL_ERROR in case of fail-on-missing=ON.
#----------------------------------------------------------------------------
macro(ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION option_name)
  ROOT_CHECK_CONNECTION("${option_name}=OFF")
  if(NO_CONNECTION)
    message(STATUS "No internet connection, disabling '${option_name}' option")
    set(${option_name} OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
  endif()
endmacro()

# Building Clad requires an internet connection, if we're not side-loading the source directory
if(clad AND NOT DEFINED CLAD_SOURCE_DIR)
  ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("clad")
endif()

#---Check for installed packages depending on the build options/components enabled --
include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)
include(ExternalProject)
include(FindPackageHandleStandardArgs)

set(lcgpackages https://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources)
string(REPLACE "-Werror " "" ROOT_EXTERNAL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

#--- Search for packages that are absolutely necessary--------------------------

#----------------------------------------------------------------------------
# ROOT_FIND_REQUIRED_DEP(PACKAGE_NAME BUILTIN_CONFIG_OPTION [MIN_REQUIRED_VERSION])
# Search for a required dependency, unless it's meant to be a built-in.
# A list of all missing required packages will be printed in case they could
# not be found as well as a hotfix to turn builtins ON.
macro(ROOT_FIND_REQUIRED_DEP PACKAGE_NAME BUILTIN_CONFIG_OPTION)
  if(NOT ${BUILTIN_CONFIG_OPTION})
    set(MIN_REQUIRED_VERSION "")
    if (${ARGC} GREATER 2) # ARGC: extra arguments + named ones
      set(MIN_REQUIRED_VERSION ${ARGV2}) # ARGV0 and ARGV1 are named required args
    endif()
    find_package(${PACKAGE_NAME} ${MIN_REQUIRED_VERSION})
    if(NOT ${PACKAGE_NAME}_FOUND)
      message(SEND_ERROR "The required package ${PACKAGE_NAME} was not found. "
      "Please install it in the system (preferred), set the corresponding CMake search variable, "
      "or opt in to downloading and auto-build it from externally provided source tarball using '-D${BUILTIN_CONFIG_OPTION}=ON'.")
      list(APPEND MISSING_PACKAGES ${PACKAGE_NAME})
      list(APPEND HOTFIX_BUILD_FLAGS '-D${BUILTIN_CONFIG_OPTION}=ON')
    endif()
  endif()
endmacro()

# Clear cache variables, or LLVM may use old values for ZLIB
# TODO: Still needed? This was ported here during a refactoring.
# When (re-)configuring cleanly (cmake --fresh), this is should be unnecessary.
foreach(suffix FOUND INCLUDE_DIR LIBRARY LIBRARY_DEBUG LIBRARY_RELEASE LIBRARIES CF)
  unset(ZLIB_${suffix} CACHE)
  unset(ZSTD_${suffix} CACHE)
endforeach()

# Request explicit user opt-in for required "easy to self-install" dependencies
# This purposely ignores the fail-on-missing=OFF behavior
ROOT_FIND_REQUIRED_DEP(LZ4 builtin_lz4)
ROOT_FIND_REQUIRED_DEP(LibLZMA builtin_lzma)
ROOT_FIND_REQUIRED_DEP(ZLIB builtin_zlib)
ROOT_FIND_REQUIRED_DEP(ZSTD builtin_zstd 1.0.0)
ROOT_FIND_REQUIRED_DEP(xxHash builtin_xxhash)
if(builtin_zlib) # We advance the builtin ZLIB creation to avoid error later in PNG/TIFF
  add_subdirectory(builtins/zlib)
endif()
if(asimage)
  ROOT_FIND_REQUIRED_DEP(GIF builtin_gif)
  ROOT_FIND_REQUIRED_DEP(JPEG builtin_jpeg)
  # PNG/TIFF must go after ZLIB builtin
  ROOT_FIND_REQUIRED_DEP(PNG builtin_png)
  ROOT_FIND_REQUIRED_DEP(TIFF builtin_tiff)
endif()
if (testing OR testsupport)
  ROOT_FIND_REQUIRED_DEP(GTest builtin_gtest 1.10)
endif()
ROOT_FIND_REQUIRED_DEP(nlohmann_json builtin_nlohmannjson 3.9)
if(unuran)
  ROOT_FIND_REQUIRED_DEP(Unuran builtin_unuran)
endif()
ROOT_FIND_REQUIRED_DEP(Freetype builtin_freetype) # needed for asimage, but also outside of it (for "graf" target)
if(opengl)
  ROOT_FIND_REQUIRED_DEP(gl2ps builtin_gl2ps)
  ROOT_FIND_REQUIRED_DEP(FTGL builtin_ftgl)
elseif(builtin_ftgl)
  message(SEND_ERROR "FTGL features enabled with \"builtin_ftgl=ON\" require \"opengl=ON\"")
  list(APPEND HOTFIX_BUILD_FLAGS '-Dopengl=ON')
endif()
foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES VERSION)
  unset(OPENSSL_${suffix} CACHE)
endforeach()
if(ssl)
  if (APPLE) # builtin OpenSSL is only supported on macOS
    ROOT_FIND_REQUIRED_DEP(OpenSSL builtin_openssl)
    if (NOT builtin_openssl)
      find_package(OpenSSL COMPONENTS SSL) # extra search for components not done before
      if(NOT OPENSSL_FOUND)
        message(SEND_ERROR "OpenSSL found but missing required component SSL. Install it on the system (preferred), or explicitly request the builtin version. Or turn off ssl option.")
        list(APPEND MISSING_PACKAGES 'OpenSSL')
        list(APPEND HOTFIX_BUILD_FLAGS '-Dssl=OFF')
      endif()
    else()
      ROOT_CHECK_CONNECTION("builtin_openssl=OFF")
      if(NO_CONNECTION)
        message(SEND_ERROR "No internet connection, disable the 'ssl' and 'builtin_openssl' options")
        list(APPEND MISSING_PACKAGES 'OpenSSL')
        list(APPEND HOTFIX_BUILD_FLAGS '-Dssl=OFF')
        list(APPEND HOTFIX_BUILD_FLAGS '-Dbuiltin_openssl=OFF')
      endif()
    endif()
  else()
    find_package(OpenSSL COMPONENTS SSL)
    if(NOT OPENSSL_FOUND)
      message(SEND_ERROR "OpenSSL found but missing required component SSL. Install it on the system (preferred), or explicitly request the builtin version. Or turn off ssl option.")
      list(APPEND MISSING_PACKAGES 'OpenSSL')
      list(APPEND HOTFIX_BUILD_FLAGS '-Dssl=OFF')
    endif()
  endif()
endif()
if(http) # Must go after SSL
  ROOT_FIND_REQUIRED_DEP(civetweb builtin_civetweb 1.15)
endif()
if(fftw3)
  ROOT_FIND_REQUIRED_DEP(FFTW builtin_fftw3)
endif()
if(vdt)
  ROOT_FIND_REQUIRED_DEP(Vdt builtin_vdt 0.4)
endif()
if(fitsio)
  ROOT_FIND_REQUIRED_DEP(CFITSIO builtin_cfitsio)
endif()
if(xrootd) # Must go after SSL
  foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
    unset(XROOTD_${suffix} CACHE)
  endforeach()
  ROOT_FIND_REQUIRED_DEP(XRootD builtin_xrootd)
  if(NOT builtin_xrootd)
    if(NOT XROOTD_FOUND)
       message(SEND_ERROR "You can also set environment variable XRDSYS to point to your XROOTD installation, "
                          "or include the installation of XROOTD in the CMAKE_PREFIX_PATH. Or turn off xrootd.")
    else()
      # XROOTD was found. Check now for required components
      foreach (component CLIENT UTILS) # ROOT requires XrdCl and XrdUtils
        if("${XROOTD_${component}_LIBRARIES}" STREQUAL "XROOTD_${component}_LIBRARIES-NOTFOUND")
          message(SEND_ERROR "XROOTD found but missing component ${component}. Install missing package on your system (preferred). "
                             "Alternatively, you can also enable the option 'builtin_xrootd' to build XROOTD internally; or turn off xrootd.")
          list(APPEND HOTFIX_BUILD_FLAGS '-Dxrootd=OFF')
        endif()
      endforeach()
    endif()
  endif()
endif()
if(builtin_xrootd)
  if(NOT ssl AND NOT builtin_openssl)
    message(SEND_ERROR "Building XRootD ('builtin_xrootd'=On) requires ssl support.")
    list(APPEND HOTFIX_BUILD_FLAGS '-Dssl=ON')
  endif()
endif()
if(xrootd AND NOT builtin_xrootd AND builtin_openssl)
  message(SEND_ERROR "Non-builtin XROOTD must not be used with builtin OpenSSL. If you want to use non-builtin XROOTD, please use the system OpenSSL")
  list(APPEND HOTFIX_BUILD_FLAGS '-Dxrootd=OFF')
endif()
if(imt)
  ROOT_FIND_REQUIRED_DEP(TBB builtin_tbb 2020)
  # Check that the found TBB does not use captured exceptions. If the header
  # <tbb/tbb_config.h> does not exist, assume that we have oneTBB newer than
  # version 2021, which does not have captured exceptions anyway.
  if(TBB_FOUND AND EXISTS "${TBB_INCLUDE_DIRS}/tbb/tbb_config.h")
    set(CMAKE_REQUIRED_INCLUDES "${TBB_INCLUDE_DIRS}")
    check_cxx_source_compiles("
#include <tbb/tbb_config.h>
#if TBB_USE_CAPTURED_EXCEPTION == 1
#error TBB uses tbb::captured_exception, not suitable for ROOT!
#endif
int main() { return 0; }" tbb_exception_result)
    if(NOT tbb_exception_result)
      message(SEND_ERROR "Found TBB uses tbb::captured_exception, not suitable for ROOT!, enable 'builtin_tbb' option or turn off 'imt'")
      list(APPEND HOTFIX_BUILD_FLAGS '-Dbuiltin_tbb=ON')
    endif()
  endif()
elseif(builtin_tbb)
  message(SEND_ERROR "TBB features enabled with \"builtin_tbb=ON\" require \"imt=ON\"")
  list(APPEND HOTFIX_BUILD_FLAGS '-Dimt=ON')
endif()

# Double package name call, needs special manual treatment, cannot call ROOT_FIND_REQUIRED_DEP:
if(NOT builtin_pcre)
  message(STATUS "Looking for PCRE")
  # Clear cache before calling find_package(PCRE),
  # necessary to be able to toggle builtin_pcre and
  # not have find_package(PCRE) find builtin pcre.
  foreach(suffix FOUND INCLUDE_DIR PCRE_LIBRARY)
    unset(PCRE_${suffix} CACHE)
  endforeach()
  find_package(PCRE2)
  if(NOT PCRE2_FOUND)
    find_package(PCRE)
    if(NOT PCRE_FOUND)
      message(SEND_ERROR "The required package PCRE2 was not found. "
      "Please install it in the system (preferred), set the corresponding CMake search variable, "
      "or opt in to downloading and auto-build it from externally provided source tarball using '-Dbuiltin_pcre=ON'.")
      list(APPEND MISSING_PACKAGES PCRE2)
      list(APPEND HOTFIX_BUILD_FLAGS '-Dbuiltin_pcre=ON')
    endif()
  endif()
endif()
if(mathmore OR (tmva-cpu AND use_gsl_cblas))
  if(builtin_gsl)
    ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("builtin_gsl")
  endif()
  message(STATUS "Looking for GSL")
  ROOT_FIND_REQUIRED_DEP(GSL builtin_gsl 1.10)
  if(NOT builtin_gsl)
    if(NOT GSL_FOUND)
      message(SEND_ERROR "GSL package not found and 'mathmore' or 'tmva-cpu' and 'use_gsl_cblas' component is required. Either disable those, or enable the option 'builtin_gsl'")
    endif()
  endif()
endif()


if(NOT "${MISSING_PACKAGES}" STREQUAL "")
  list(REMOVE_DUPLICATES MISSING_PACKAGES)
  message(SEND_ERROR "The following packages need to be installed system-wide to build ROOT: ${MISSING_PACKAGES}")
endif()
if(NOT "${HOTFIX_BUILD_FLAGS}" STREQUAL "")
  list(REMOVE_DUPLICATES HOTFIX_BUILD_FLAGS)
  set(HOTFIX_BUILD_FLAGS_MESSAGE "Alternatively, a hotfix would be to add these flags to your CMake call:\n")

  foreach(_item IN LISTS HOTFIX_BUILD_FLAGS)
    string(APPEND HOTFIX_BUILD_FLAGS_MESSAGE "  ${_item} \\\n")
  endforeach()

  # Remove final trailing backslash and newline
  string(REGEX REPLACE "\\\\\n$" "" HOTFIX_BUILD_FLAGS_MESSAGE "${HOTFIX_BUILD_FLAGS_MESSAGE}")

  message(FATAL_ERROR "${HOTFIX_BUILD_FLAGS_MESSAGE}")
endif()

#---On MacOSX, try to find frameworks after standard libraries or headers------------
set(CMAKE_FIND_FRAMEWORK LAST)

#---If -Dshared=Off, prefer static libraries-----------------------------------------
if(NOT shared)
  if(WINDOWS)
    message(FATAL_ERROR "Option \"shared=Off\" not supported on Windows!")
  else()
    message("Preferring static libraries.")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;${CMAKE_FIND_LIBRARY_SUFFIXES}")
  endif()
endif()

#---Check for Zlib ------------------------------------------------------------------
if(builtin_zlib)
  # add_subdirectory(builtins/zlib) Already done above to prevent conflicts
else()
  # If not built-in, check if this is zlib-ng
  set(CMAKE_REQUIRED_INCLUDES ${ZLIB_INCLUDE_DIRS})
  message(STATUS "Checking whether zlib-ng is provided")
  check_c_source_compiles("
      #include <zlib.h>
      #ifndef ZLIBNG_VERNUM
      #error Not zlib-ng
      #endif
      int main() { return 0; }
  " ZLIB_NG)
endif()

if(ZLIB_NG)
  message(STATUS "Zlib-ng detected")
else()
  message(STATUS "Zlib detected")
endif()

#---Check for nlohmann/json.hpp---------------------------------------------------------
if(NOT builtin_nlohmannjson)
  # ROOTEve wants to know if it comes with json_fwd.hpp:
  if(TARGET nlohmann_json::nlohmann_json)
    get_target_property(inc_dirs nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
    foreach(dir ${inc_dirs})
      if(EXISTS "${dir}/nlohmann/json_fwd.hpp")
        target_compile_definitions(nlohmann_json::nlohmann_json INTERFACE NLOHMANN_JSON_PROVIDES_FWD_HPP)
      endif()
    endforeach()
  endif()
else()
  add_subdirectory(builtins/nlohmann)
endif()

#---Check for Unuran ------------------------------------------------------------------
if (builtin_unuran)
  add_subdirectory(builtins/unuran)
endif()

#---Check for Freetype---------------------------------------------------------------
if(builtin_freetype)
  add_subdirectory(builtins/freetype)
elseif(NOT Freetype_VERSION AND FREETYPE_VERSION_STRING)
  # on mac brew installed freetype version_string is returned
  message(STATUS "Found legacy freetype ${FREETYPE_VERSION_STRING}")
  set(Freetype_VERSION ${FREETYPE_VERSION_STRING})
endif()

#---Check for Cocoa/Quartz graphics backend (MacOS X only)---------------------------
# Note that this check happens *after* the above check for FreeType because that
# library is needed for builds on Apple with Cocoa graphics
if(cocoa)
  if(APPLE)
    set(x11 OFF CACHE BOOL "Disabled because cocoa requested (${x11_description})" FORCE)
  else()
    message(STATUS "Cocoa option can only be enabled on MacOSX platform")
    set(cocoa OFF CACHE BOOL "Disabled because only available on MacOSX (${cocoa_description})" FORCE)
  endif()
endif()

#---Check for PCRE-------------------------------------------------------------------
if(builtin_pcre)
  add_subdirectory(builtins/pcre)
endif()

#---Check for LZMA-------------------------------------------------------------------
if(builtin_lzma)
  add_subdirectory(builtins/lzma)
endif()

#---Check for xxHash-----------------------------------------------------------------
if(builtin_xxhash)
  add_subdirectory(builtins/xxhash)
endif()

#---Check for ZSTD-------------------------------------------------------------------
if(builtin_zstd)
  add_subdirectory(builtins/zstd)
endif()

#---Check for LZ4--------------------------------------------------------------------
if(builtin_lz4)
  add_subdirectory(builtins/lz4)
endif()

#---Check for X11 which is mandatory lib on Unix--------------------------------------
if(x11)
  message(STATUS "Looking for X11")
  if(X11_X11_INCLUDE_PATH)
    set(X11_FIND_QUIETLY 1)
  endif()
  find_package(X11 REQUIRED COMPONENTS Xpm Xft Xext)
  list(REMOVE_DUPLICATES X11_INCLUDE_DIR)
  if(NOT X11_FIND_QUIETLY)
    message(STATUS "X11_INCLUDE_DIR: ${X11_INCLUDE_DIR}")
    message(STATUS "X11_LIBRARIES: ${X11_LIBRARIES}")
    message(STATUS "X11_Xpm_INCLUDE_PATH: ${X11_Xpm_INCLUDE_PATH}")
    message(STATUS "X11_Xpm_LIB: ${X11_Xpm_LIB}")
    message(STATUS "X11_Xft_INCLUDE_PATH: ${X11_Xft_INCLUDE_PATH}")
    message(STATUS "X11_Xft_LIB: ${X11_Xft_LIB}")
    message(STATUS "X11_Xext_INCLUDE_PATH: ${X11_Xext_INCLUDE_PATH}")
    message(STATUS "X11_Xext_LIB: ${X11_Xext_LIB}")
  endif()
endif()

#---Check for all kind of graphics includes needed by libAfterImage--------------------
if(asimage)
  if(NOT x11 AND NOT cocoa AND NOT WIN32)
    message(STATUS "Switching off 'asimage' because neither 'x11' nor 'cocoa' are enabled")
    set(asimage OFF CACHE BOOL "Disabled because neither x11 nor cocoa are enabled (${asimage_description})" FORCE)
  endif()
endif()
if(asimage)

  if(builtin_gif)
    add_subdirectory(builtins/libgif)
    get_target_property(GIF_INCLUDE_DIR GIF::GIF INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(GIF_LIBRARY_LOCATION GIF::GIF IMPORTED_LOCATION)
  endif()
  list(APPEND ASEXTRA_LIBRARIES GIF::GIF)

  if(builtin_png)
    add_subdirectory(builtins/libpng)
    get_target_property(PNG_INCLUDE_DIR PNG::PNG INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(PNG_LIBRARY_LOCATION PNG::PNG IMPORTED_LOCATION)
  endif()
  list(APPEND ASEXTRA_LIBRARIES PNG::PNG)

  if(builtin_jpeg)
    add_subdirectory(builtins/libjpeg)
    get_target_property(JPEG_INCLUDE_DIR JPEG::JPEG INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(JPEG_LIBRARY_LOCATION JPEG::JPEG IMPORTED_LOCATION)
  endif()
  list(APPEND ASEXTRA_LIBRARIES JPEG::JPEG)

  if(builtin_tiff)
    add_subdirectory(builtins/libtiff)
    get_target_property(TIFF_INCLUDE_DIR TIFF::TIFF INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(TIFF_LIBRARY_LOCATION TIFF::TIFF IMPORTED_LOCATION)
  endif()
  list(APPEND ASEXTRA_LIBRARIES TIFF::TIFF)

  add_subdirectory(builtins/libAfterImage) # It's a hard-coded builtin, was forked in 2008, system-wide version misses many patches
endif()

#---Check for Python installation-------------------------------------------------------
message(STATUS "Looking for Python")
# On macOS, prefer user-provided Pythons.
set(Python3_FIND_FRAMEWORK LAST)

# Even if we don't build PyROOT, one still need python executable to run some scripts
list(APPEND python_components Interpreter)
if(pyroot AND NOT (tpython OR tmva-pymva))
  # We have to only look for the Python development module in order to be able to build ROOT with a pip backend
  # In particular, it is forbidden to link against libPython.so, see https://peps.python.org/pep-0513/#libpythonx-y-so-1
  list(APPEND python_components Development.Module)
elseif(tpython OR tmva-pymva)
  list(APPEND python_components Development)
endif()
if(tmva-pymva)
  list(APPEND python_components NumPy)
endif()
find_package(Python3 3.10 COMPONENTS ${python_components})

# Detect whether the found Python interpreter is a free-threaded build
# (Py_GIL_DISABLED is defined in pyconfig.h). The limited C API is not
# supported in free-threaded builds; including Python.h with Py_LIMITED_API
# defined produces a hard error there
# (https://docs.python.org/3/howto/free-threading-extensions.html).
# Checking the preprocessor symbol directly is more reliable than asking the
# interpreter (e.g. sysconfig.get_config_var may misreport, and
# sys._is_gil_enabled() can be overridden at runtime via PYTHON_GIL=1).
set(Python3_GIL_DISABLED FALSE)
if(Python3_Development_FOUND OR Python3_Development.Module_FOUND)
  include(CheckCXXSourceCompiles)
  set(_old_required_includes ${CMAKE_REQUIRED_INCLUDES})
  set(CMAKE_REQUIRED_INCLUDES ${Python3_INCLUDE_DIRS})
  check_cxx_source_compiles("
    #include <Python.h>
    #ifndef Py_GIL_DISABLED
    #error \"GIL is not disabled\"
    #endif
    int main() { return 0; }
  " ROOT_PYTHON_GIL_DISABLED)
  set(CMAKE_REQUIRED_INCLUDES ${_old_required_includes})
  if(ROOT_PYTHON_GIL_DISABLED)
    set(Python3_GIL_DISABLED TRUE)
    message(STATUS "Python ${Python3_VERSION} is a free-threaded build (Py_GIL_DISABLED defined); the limited C API will not be used")
  endif()
endif()

#---Check for OpenGL installation-------------------------------------------------------
# OpenGL is required by various graf3d features that are enabled with opengl=ON,
# or by the Cocoa-related code that always requires it.
if(opengl OR cocoa)
  message(STATUS "Looking for OpenGL")
  if(APPLE)
    set(CMAKE_FIND_FRAMEWORK FIRST)
    find_package(OpenGL)
    set(CMAKE_FIND_FRAMEWORK LAST)
  else()
    find_package(OpenGL)
  endif()
  if(NOT OPENGL_FOUND OR NOT OPENGL_GLU_FOUND)
    if(fail-on-missing)
      message(SEND_ERROR "OpenGL package (with GLU) not found and opengl option required")
    elseif(cocoa)
      message(FATAL_ERROR "OpenGL package (with GLU) not found and opengl option required for \"cocoa=ON\"")
    else()
      message(STATUS "OpenGL (with GLU) not found. Switching off opengl option")
      set(opengl OFF CACHE BOOL "Disabled because OpenGL (with GLU) not found (${opengl_description})" FORCE)
    endif()
  endif()
endif()
# OpenGL should be working only with x11 (Linux),
# in case when -Dall=ON -Dx11=OFF, we will just disable opengl.
if(NOT WIN32 AND NOT APPLE)
  if(opengl AND NOT x11)
    message(STATUS "OpenGL was disabled, since it is requires x11 on Linux")
    set(opengl OFF CACHE BOOL "OpenGL requires x11" FORCE)
  endif()
endif()
# The opengl flag enables the graf3d features that depend on OpenGL, and these
# features also depend on asimage. Therefore, the configuration will fail if
# asimage is off. See also: https://github.com/root-project/root/issues/16250
if(opengl AND NOT asimage)
  message(SEND_ERROR "OpenGL features enabled with \"opengl=ON\" require \"asimage=ON\"")
endif()

#---Check for gl2ps ------------------------------------------------------------------
if(opengl AND builtin_gl2ps)
  add_subdirectory(builtins/gl2ps)
endif()

#---Check for Graphviz installation-------------------------------------------------------
if(gviz)
  message(STATUS "Looking for Graphviz")
  find_package(Graphviz)
  if(NOT GRAPHVIZ_FOUND)
    message(SEND_ERROR "Graphviz libraries not found while -Dgviz=On.")
  endif()
endif()

#---Check for XML Parser Support-----------------------------------------------------------
if(xml)
  message(STATUS "Looking for LibXml2")
  find_package(LibXml2)
  if(NOT LIBXML2_FOUND)
    if(fail-on-missing)
      message(SEND_ERROR "LibXml2 libraries not while -Dxml=ON")
    else()
      message(STATUS "LibXml2 not found. Switching off xml option")
      set(xml OFF CACHE BOOL "Disabled because LibXml2 not found (${xml_description})" FORCE)
    endif()
  endif()
endif()

#---Check for OpenSSL------------------------------------------------------------------
if(builtin_openssl)
  add_subdirectory(builtins/openssl)
endif()

#---Check for FastCGI-----------------------------------------------------------
if(fcgi)
  message(STATUS "Looking for FastCGI")
  find_package(FastCGI)
  if(NOT FASTCGI_FOUND)
    message(SEND_ERROR "FastCGI library not found while -Dfcgi=On")
  endif()
endif()

#--- Check for civetweb - (has to go after SSL) ---------------------------------------
if(http AND NOT builtin_civetweb)
  if(civetweb_FOUND)
    try_compile(CIVETWEB_FEATURE_API
      SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/civetweb_check_features.c"
      LINK_LIBRARIES civetweb::civetweb
      OUTPUT_VARIABLE CIVETWEB_FEATURE_API_LOG
    )
    if (CIVETWEB_FEATURE_API)
      try_run(RUN_RESULT COMPILE_RESULT
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/civetweb_check_features.c"
        LINK_LIBRARIES civetweb::civetweb
        COMPILE_OUTPUT_VARIABLE BUILD_LOG
        RUN_OUTPUT_VARIABLE CIVETWEB_FEATURES
      )
      if(COMPILE_RESULT)
        message(STATUS "Detected civetweb feature mask: ${CIVETWEB_FEATURES}")
      else()
        message(FATAL_ERROR "Could not run civetweb features: ${BUILD_LOG}")
      endif()
      math(EXPR CIVETWEB_HAS_WEBSOCKET "(${CIVETWEB_FEATURES} >> 4) & 0x1")
      math(EXPR CIVETWEB_HAS_ZLIB "(${CIVETWEB_FEATURES} >> 9) & 0x1")
      math(EXPR CIVETWEB_HAS_X_DOM_SOCKET "(${CIVETWEB_FEATURES} >> 11) & 0x1")
      message(STATUS "civetweb websocket ; zlib ; xdomsocket support: ${CIVETWEB_HAS_WEBSOCKET} ; ${CIVETWEB_HAS_ZLIB} ; ${CIVETWEB_HAS_X_DOM_SOCKET}")
    else()
      message(FATAL_ERROR "Could not check for civetweb features: ${CIVETWEB_FEATURE_API_LOG}")
    endif()

    if(NOT "${CIVETWEB_HAS_WEBSOCKET}" STREQUAL "1" OR NOT "${CIVETWEB_HAS_ZLIB}" STREQUAL "1" OR NOT "${CIVETWEB_HAS_X_DOM_SOCKET}" STREQUAL "1")
      # Clear cache vars by find_package system-civetweb
      foreach(var CIVETWEB_LIBRARIES CIVETWEB_LIBRARY CIVETWEB_LIBRARY_DEBUG CIVETWEB_LIBRARY_RELEASE CIVETWEB_FOUND CIVETWEB_VERSION CIVETWEB_INCLUDE_DIR CIVETWEB_LIBRARY CIVETWEB_LIBRARIES)
        unset(${var})
        unset(${var} CACHE)
      endforeach()
      message(SEND_ERROR "System-wide civetweb found but does not include websocket or zlib or xdomsocket components (-DCIVETWEB_ENABLE_WEBSOCKETS=ON -DCIVETWEB_ENABLE_ZLIB=ON -DCIVETWEB_ENABLE_X_DOM_SOCKET=ON). Set `-Dbuiltin_civetweb=ON` as workaround or switch `-Dhttp=OFF`.")
    endif()
  endif()
endif()
if(http AND builtin_civetweb)
  add_subdirectory(builtins/civetweb)
endif()

#---Check for SQLite-------------------------------------------------------------------
if(sqlite)
  message(STATUS "Looking for SQLite")
  find_package(Sqlite)
  if(NOT SQLITE_FOUND)
    if(fail-on-missing)
      message(SEND_ERROR "SQLite libraries not found while -Dsqlite=ON")
    else()
      message(STATUS "SQLite not found. Switching off sqlite option")
      set(sqlite OFF CACHE BOOL "Disabled because SQLite not found (${sqlite_description})" FORCE)
    endif()
  endif()
endif()

#---Check for Pythia8-------------------------------------------------------------------
if(pythia8)
  message(STATUS "Looking for Pythia8")
  find_package(Pythia8)
  if(NOT PYTHIA8_FOUND)
    message(SEND_ERROR "Pythia8 libraries not found while -Dpythia8=ON")
  endif()
endif()

#---Check for FFTW3-------------------------------------------------------------------
if(builtin_fftw3)
  ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("builtin_fftw3")
endif()
if(builtin_fftw3)
  add_subdirectory(builtins/fftw3)
  set(fftw3 ON CACHE BOOL "Enabled because builtin_fftw3 requested (${fftw3_description})" FORCE)
endif()

#---Check for fitsio-------------------------------------------------------------------
if(fitsio OR builtin_cfitsio)
  if(builtin_cfitsio)
    ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("builtin_cfitsio")
  endif()
  if(builtin_cfitsio)
    add_library(CFITSIO::CFITSIO STATIC IMPORTED GLOBAL)
    add_subdirectory(builtins/cfitsio)
    if(NOT fitsio)
      set(fitsio ON CACHE BOOL "Enabled because builtin_cfitsio requested (${fitsio_description})" FORCE)
    endif()
  endif()
endif()

#---Check Shadow password support----------------------------------------------------
if(shadowpw)
  if(NOT EXISTS /etc/shadow)  #---TODO--The test always succeeds because the actual file is protected
    if(NOT CMAKE_SYSTEM_NAME MATCHES Linux)
      message(STATUS "Support Shadow password not found. Switching off shadowpw option")
      set(shadowpw OFF CACHE BOOL "Disabled because /etc/shadow not found (${shadowpw_description})" FORCE)
    endif()
  endif()
endif()

#---Configure Xrootd support---------------------------------------------------------
if(xrootd AND NOT builtin_xrootd)
  if(XRootD_VERSION VERSION_LESS 5.8.4)
    # Remove -D from XRootD's exported compile definitions. https://github.com/xrootd/xrootd/issues/2543
    foreach(XRDTarget XRootD::XrdCl XRootD::XrdUtils)
      if(TARGET ${XRDTarget})
        get_target_property(PROP ${XRDTarget} INTERFACE_COMPILE_DEFINITIONS)
        list(TRANSFORM PROP REPLACE "^-D" "")
        set_property(TARGET ${XRDTarget} PROPERTY INTERFACE_COMPILE_DEFINITIONS ${PROP})
      endif()
    endforeach()
  endif()
endif()

if(builtin_xrootd)
  ROOT_CHECK_CONNECTION("builtin_xrootd=OFF")
  if(NO_CONNECTION)
    message(SEND_ERROR "No internet connection. Please check your connection, or disable the 'builtin_xrootd'"
      " option")
  endif()
  add_subdirectory(builtins/xrootd)
  set(xrootd ON CACHE BOOL "Enabled because builtin_xrootd requested (${xrootd_description})" FORCE)
endif()

# Backward compatibility for XRootD <v5.8 without CMake targets:
if(xrootd AND NOT TARGET XRootD::XrdCl)
  # Before v5.7.0, XROOTD_INCLUDE_DIRS includes private headers, like:
  #   <xrootd_include_dir>;<xrootd_include_dir>/private
  # The private headers are not always installed, so the configure step might fail.
  # ROOT doesn't need these headers, so it's best to remove them.
  list(FILTER XROOTD_INCLUDE_DIRS EXCLUDE REGEX .*/private)

  add_library(XRootD::XrdCl SHARED IMPORTED)
  set_target_properties(XRootD::XrdCl PROPERTIES IMPORTED_LOCATION ${XROOTD_CLIENT_LIBRARIES})
  target_include_directories(XRootD::XrdCl SYSTEM INTERFACE $<BUILD_INTERFACE:${XROOTD_INCLUDE_DIRS}>)

  add_library(XRootD::XrdUtils SHARED IMPORTED)
  set_target_properties(XRootD::XrdUtils PROPERTIES IMPORTED_LOCATION ${XROOTD_UTILS_LIBRARIES})
endif()

#---Check for Apache Arrow
if(arrow)
  find_package(Arrow)
  if(NOT ARROW_FOUND)
    message(SEND_ERROR "Apache Arrow not found but is required. Please set ARROW_ROOT to point to your Arrow installation, "
                          "or include the installation of Arrow in the CMAKE_PREFIX_PATH. Or disable option 'arrow'.")
  endif()
endif()

#---Check for dCache-------------------------------------------------------------------
if(dcache)
  find_package(DCAP)
  if(NOT DCAP_FOUND)
    message(SEND_ERROR "dCap library not found while -Ddcache=ON"
      " Set variable DCAP_ROOT to point to your dCache installation. Or disable option 'dcache'.")
  endif()
endif()

#---Check for ftgl if needed----------------------------------------------------------
if(opengl AND builtin_ftgl)
  add_subdirectory(builtins/ftgl)
endif()

#---Check for Davix library-----------------------------------------------------------
foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
  unset(DAVIX_${suffix} CACHE)
endforeach()

if(davix)
  if(MSVC)
    message(FATAL_ERROR "Davix is not supported on Windows")
  endif()

  if(fail-on-missing)
    find_package(Davix 0.6.4 REQUIRED)
    if(DAVIX_VERSION VERSION_GREATER_EQUAL 0.6.8 AND DAVIX_VERSION VERSION_LESS 0.7.1)
      message(WARNING "Davix versions 0.6.8 to 0.7.0 have a bug and do not work with ROOT, please upgrade to 0.7.1 or later.")
    endif()
  else()
    find_package(Davix 0.6.4)
    if(NOT DAVIX_FOUND)
      message(STATUS "Davix not found. Switching off davix option")
      set(davix OFF CACHE BOOL "Disabled because dependencies not found (${davix_description})" FORCE)
    endif()
  endif()
endif()

#---Check for curl library-----------------------------------------------------------
foreach(suffix FOUND INCLUDE_DIR INCLUDE_DIRS LIBRARY LIBRARIES)
  unset(CURL_${suffix} CACHE)
endforeach()

if(curl)
  message(STATUS "Looking for libcurl")
  if(MSVC)
    # On Windows, we must initialize libcurl lazily (not in a static [DLL] initializer), and this
    # works only safely as of 7.84 with the threadsafe option
    find_package(CURL 7.84 COMPONENTS HTTP HTTPS threadsafe)
  else()
    # Matches the libcurl version on EL9/10
    find_package(CURL 7.76 COMPONENTS HTTP HTTPS)
  endif()

  if(NOT CURL_FOUND)
    if(fail-on-missing)
      message(SEND_ERROR "libcurl not found and curl option required")
    else()
      message(STATUS "libcurl not found. Switching off curl option")
      set(curl OFF CACHE BOOL "Disabled because libcurl was not found (${curl_description})" FORCE)
    endif()
  endif()
endif()

#---Check for liburing----------------------------------------------------------------
if (uring)
  if(NOT CMAKE_SYSTEM_NAME MATCHES Linux)
    set(uring OFF CACHE BOOL "Disabled because liburing is only available on Linux" FORCE)
    message(STATUS "liburing was disabled because it is only available on Linux")
  else()
    message(STATUS "Looking for liburing")
    find_package(liburing)
    if(NOT LIBURING_FOUND)
      message(SEND_ERROR "liburing not found and uring option required. Install it on the system or disable option 'uring'.")
    endif()
  endif()
endif()

#---Check for DAOS----------------------------------------------------------------
if (daos AND daos_mock)
  message(FATAL_ERROR "Options `daos` and `daos_mock` are mutually exclusive; only one of them should be specified.")
endif()
if (testing AND NOT daos AND NOT WIN32)
  set(daos_mock ON CACHE BOOL "Enable `daos_mock` if `testing` option was set" FORCE)
endif()

if (daos OR daos_mock)
  find_package(libuuid)
  if(NOT libuuid_FOUND)
    message(SEND_ERROR "libuuid not found and it is required (daos or daos_mock option enabled). Install it on the system, or disable options 'daos' and 'daos_mock'")
  endif()
endif()
if (daos)
  find_package(DAOS)
  if(NOT DAOS_FOUND)
    message(SEND_ERROR "libdaos not found while -Ddaos=ON. Install it on the system, or disable option 'daos'")
  endif()
endif()

#---Check for TBB---------------------------------------------------------------------
if(imt AND NOT builtin_tbb)
  if(MSVC)
    set(TBB_CXXFLAGS "-D__TBB_NO_IMPLICIT_LINKAGE=1 -DTBB_SUPPRESS_DEPRECATED_MESSAGES=1")
  else()
    set(TBB_CXXFLAGS "-DTBB_SUPPRESS_DEPRECATED_MESSAGES=1")
  endif()
endif()

if(builtin_tbb)
  ROOT_CHECK_CONNECTION("builtin_tbb=OFF")
  if(NO_CONNECTION)
    message(STATUS "No internet connection, disabling 'builtin_tbb' and 'imt' options")
    set(builtin_tbb OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    set(imt OFF CACHE BOOL "Disabled because 'builtin_tbb' was set but there is no internet connection" FORCE)
  endif()
endif()

if(builtin_tbb)
  add_subdirectory(builtins/tbb)
endif()

if(builtin_vdt)
  ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("builtin_vdt")
endif()

#---Check for Vdt--------------------------------------------------------------------
if(vdt OR builtin_vdt)
  if(builtin_vdt)
    add_subdirectory(builtins/vdt)
  endif()
endif()

#---Check for VecGeom--------------------------------------------------------------------
if (vecgeom)
  message(STATUS "Looking for VecGeom")
  find_package(VecGeom 1.2 CONFIG)
  if(NOT VecGeom_FOUND)
    message(SEND_ERROR "VecGeom not found. Ensure that the installation of VecGeom is in the CMAKE_PREFIX_PATH, or disable 'vecgeom'")
  else()
    message(STATUS "   Found VecGeom " ${VecGeom_VERSION})
  endif()
endif()

if(experimental_adaptivecpp)
  # Building adaptivecpp requires an internet connection, if we're not side-loading the source directory
  if(NOT DEFINED ADAPTIVECPP_SOURCE_DIR)
    ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("experimental_adaptivecpp")
  endif()
  include(SetupAdaptiveCpp)

  add_compile_definitions(CLING_WITH_ADAPTIVECPP)

  set(HIPSYCL_NO_FIBERS ON)
  set(WITH_OPENCL_BACKEND OFF)
  set(WITH_LEVEL_ZERO_BACKEND OFF)

  find_package(AdaptiveCpp)
  if (AdaptiveCpp_FOUND)
    set(sycl ON)
    set(SYCL_COMPILER_FLAGS "-ffast-math ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}")
    message(STATUS "SYCL compiler flags: ${SYCL_COMPILER_FLAGS}")
    separate_arguments(SYCL_COMPILER_FLAGS NATIVE_COMMAND ${SYCL_COMPILER_FLAGS})
    function(add_sycl_to_root_target)
      CMAKE_PARSE_ARGUMENTS(ARG "" "TARGET" "SOURCES" "COMPILE_DEFINITIONS" ${ARGN})
      add_dependencies(${ARG_TARGET} acpp-rt)
      add_sycl_to_target(TARGET ${ARG_TARGET} SOURCES ${ARG_SOURCES})
      target_link_libraries(${ARG_TARGET} INTERFACE AdaptiveCpp::acpp-rt)
      target_compile_options(${ARG_TARGET} PUBLIC ${SYCL_COMPILER_FLAGS})
      target_compile_definitions(${ARG_TARGET} PUBLIC ${ARG_COMPILE_DEFINITIONS})
    endfunction()
    message(STATUS "AdaptiveCpp sycl enabled")
  else()
    message(SEND_ERROR "AdaptiveCpp library not found, install it or disable 'experimental_adaptivecpp'")
  endif()
endif()

#---Check for optional TMVA-SOFIE testing dependency (BLAS)-------------------------------
# SOFIE itself has no external dependencies: ONNX models are read with a small
# self-contained protobuf wire-format decoder (tmva/sofie_parsers/src/onnx.hxx).

if(tmva AND testing AND test_tmva_sofie)
  message(STATUS "Looking for BLAS as an optional testing dependency of TMVA-SOFIE")
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    message(SEND_ERROR "BLAS not found, but it's required for TMVA-SOFIE testing. Please install BLAS or configure with test_tmva_sofie=OFF")
  endif()
endif()

#---Figure out if TMVA CPU should be built and which BLAS we will use ------------------
if(tmva-cpu)
  if (NOT tmva)
    set(tmva-cpu   OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-cpu_description})" FORCE)
  elseif(NOT imt)
    set(tmva-cpu OFF CACHE BOOL "Disabled because 'imt' is disabled (${tmva-cpu_description})" FORCE)
  endif()
endif()
if(tmva-cpu)
  if (NOT use_gsl_cblas)
    find_package(BLAS)
    if(NOT BLAS_FOUND)
      # If no optimized BLAS library was found, we fall back to attempting to
      # use the GSL CBLAS. If ROOT is built with fail-on-missing=ON, this
      # usually means that the user does not want us to change build flags
      # automatically, so we send an error.
      if(fail-on-missing)
        message(SEND_ERROR "Option tmva-cpu requires a BLAS library, but none could be found on the system. Either install a BLAS library like OpenBLAS (preferred), or set use_gsl_cblas=ON (possibly also builtin_gsl=ON if GSL not installed on the system).")
      else()
        set(use_gsl_cblas ON CACHE BOOL "Auto-enabling GSL CBLAS for TMVA [GPL]" FORCE)
      endif()
    endif()
  endif()
endif()

#---Check for GSL library---------------------------------------------------------------
if(mathmore OR builtin_gsl OR (tmva-cpu AND use_gsl_cblas))
  if(builtin_gsl)
    ROOT_CHECK_CONNECTION_AND_DISABLE_OPTION("builtin_gsl")
  endif()
  if(builtin_gsl)
    add_subdirectory(builtins/gsl)
  endif()
endif()

#---TMVA and its dependencies------------------------------------------------------------
if(tmva-cpu)
  # ROOT internal BLAS target for TMVA
  add_library(Blas INTERFACE)
  add_library(ROOT::BLAS ALIAS Blas)
  if (NOT use_gsl_cblas AND BLAS_FOUND)
    target_link_libraries(Blas INTERFACE BLAS::BLAS)
  elseif(use_gsl_cblas AND GSL_FOUND)
    if (builtin_gsl)
      message(STATUS "Using builtin GSL CBLAS for optional parts of TMVA")
    else()
      message(STATUS "Using system GSL CBLAS for optional parts of TMVA")
    endif()
    target_compile_definitions(Blas INTERFACE -DR__USE_CBLAS)
    target_link_libraries(Blas INTERFACE GSL::gslcblas)
  else()
    if(fail-on-missing)
      message(SEND_ERROR "tmva-cpu can't be built because BLAS was not found!")
    else()
      message(STATUS "tmva-cpu disabled because BLAS was not found")
      set(tmva-cpu OFF CACHE BOOL "Disabled because BLAS was not found (${tmva-cpu_description})" FORCE)
    endif()
  endif()
endif()
if(tmva)
  if(tmva-gpu AND NOT CMAKE_CUDA_COMPILER)
    set(tmva-gpu OFF CACHE BOOL "Disabled because cuda not found" FORCE)
  endif()
  if(tmva-gpu)
    # So far, TMVA is the only package that uses the CUDA toolkit. RooFit is
    # just compiling libraries with the NVidia compiler itself. If more ROOT
    # components depend on the CUDA toolkit, this should be moved.
    find_package(CUDAToolkit REQUIRED)

    ### Look for package CuDNN.
    if (tmva-cudnn)
      find_package(CUDNN)
      if (CUDNN_FOUND)
        message(STATUS "CuDNN library found: " ${CUDNN_LIBRARIES})
        # Once proper cuDNN support in CMake, replace this with an alias target:
        add_library(ROOT::cuDNN SHARED IMPORTED)
        set_property(TARGET ROOT::cuDNN PROPERTY IMPORTED_LOCATION ${CUDNN_LIBRARIES})
        target_include_directories(ROOT::cuDNN INTERFACE ${CUDNN_INCLUDE_DIR})
      else()
        message(SEND_ERROR "cudnn not found  while -Dtmva-cudnn=ON. Install it on the system, or disable option 'tmva-cudnn'")
      endif()
    endif()
  endif()
  if(tmva-pymva)
    if(NOT Python3_NumPy_FOUND OR NOT Python3_Development_FOUND)
      message(SEND_ERROR "TMVA: numpy python package or Python development package not found and tmva-pymva component required"
                          " (python executable: ${Python3_EXECUTABLE})")
    endif()
  endif()
else()
  set(tmva-gpu   OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-gpu_description})"   FORCE)
  set(tmva-cudnn OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-cudnn_description})"  FORCE)
  set(tmva-pymva OFF CACHE BOOL "Disabled because 'tmva' is disabled (${tmva-pymva_description})" FORCE)
endif(tmva)

#---Check for PyROOT---------------------------------------------------------------------
if(pyroot)

  if(Python3_Development.Module_FOUND)
    message(STATUS "PyROOT: development package found. Building for version ${Python3_VERSION}")
  else()
    if(fail-on-missing)
      message(SEND_ERROR "PyROOT: Python development package not found and pyroot component required"
                          " (python executable: ${Python3_EXECUTABLE})")
    else()
      message(STATUS "PyROOT: Python development package not found for python ${Python3_EXECUTABLE}. Switching off pyroot option")
      set(pyroot OFF CACHE BOOL "Disabled because Python development package was not found for ${Python3_EXECUTABLE}" FORCE)
    endif()
  endif()

endif()

#---Check for TPython---------------------------------------------------------------------
if(tpython)

  if(NOT Python3_Development_FOUND)
    if(fail-on-missing)
      message(SEND_ERROR "TPython: Python development package not found and tpython component required"
                          " (python executable: ${Python3_EXECUTABLE})")
    else()
      message(STATUS "TPython: Python development package not found for python ${Python3_EXECUTABLE}. Switching off tpython option")
      set(tpython OFF CACHE BOOL "Disabled because Python development package was not found for ${Python3_EXECUTABLE}" FORCE)
    endif()
  endif()

endif()

#---Check for MPI---------------------------------------------------------------------
if (mpi)
  message(STATUS "Looking for MPI")
  find_package(MPI)
  if(NOT MPI_FOUND)
    message(SEND_ERROR "MPI not found. Ensure that the installation of MPI is in the CMAKE_PREFIX_PATH."
      " Example: CMAKE_PREFIX_PATH=<MPI_install_path> (e.g. \"/usr/local/mpich\"). Or disable option 'mpi'")
  endif()
endif()

#---Check for ZeroMQ when building RooFit::MultiProcess--------------------------------------------

if (roofit_multiprocess)
    message(STATUS "Looking for ZeroMQ (libzmq)")

    # Temporarily prefer config mode over module mode, so that a CMake-installed system version
    # gets detected before looking for an autotools-installed system version (which the
    # FindZeroMQ.cmake module does).
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE ${CMAKE_FIND_PACKAGE_PREFER_CONFIG})
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    # The fail-on-missing branching is not implemented, and we always look for
    # ZeroMQ and cppzmq with REQUIRED to fail configuration if not available.
    # That's because the roofit_multiprocess option can only be deliberately
    # enabled by the user with roofit_multiprocess=ON, in which case it would
    # be frustrating to get it auto-disabled on missing dependencies.
    find_package(ZeroMQ 4.3.5 REQUIRED)

    # Reset default find_package mode
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ${CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE})
    unset(CMAKE_FIND_PACKAGE_PREFER_CONFIG_ORIGINAL_VALUE)

    message(STATUS "Looking for ZeroMQ C++ bindings (cppzmq)")
    find_package(cppzmq REQUIRED)
endif (roofit_multiprocess)

#---Check for googletest---------------------------------------------------------------
if (testing OR testsupport)
  if (builtin_gtest)
    ROOT_CHECK_CONNECTION("testing=OFF")
    if(NO_CONNECTION)
      message(STATUS "No internet connection, disabling the 'testing', 'testsupport' and 'builtin_gtest' options")
      set(testing OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      set(testsupport OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
      set(builtin_gtest OFF CACHE BOOL "Disabled because there is no internet connection" FORCE)
    else()
      add_subdirectory(builtins/gtest)
    endif()
  endif()
endif()

if (testing OR testsupport)
  # Verify that all GTest subcomponents are installed
  foreach(LIBNAME gtest_main gmock_main gtest gmock)
    if(NOT TARGET GTest::${LIBNAME} AND NOT TARGET ${LIBNAME})
      message(SEND_ERROR "Missing installation of GTest subcomponent ${LIBNAME}")
    endif()
  endforeach()
  # Starting from cmake 3.23, the GTest targets will have stable names.
  # ROOT was updated to use those, but for older CMake versions, we have to declare the aliases:
  foreach(LIBNAME gtest_main gmock_main gtest gmock)
    if(NOT TARGET GTest::${LIBNAME} AND TARGET ${LIBNAME})
      add_library(GTest::${LIBNAME} ALIAS ${LIBNAME})
    endif()
  endforeach()
endif()

#------------------------------------------------------------------------------------
if(webgui)
  if(NOT "$ENV{OPENUI5DIR}" STREQUAL "" AND EXISTS "$ENV{OPENUI5DIR}/resources/sap-ui-core.js")
     # create symbolic link on existing openui5 installation
     # should be used only for debug purposes to be able try different openui5 version
     # cannot be used for installation purposes
     message(STATUS "openui5 - use from $ENV{OPENUI5DIR}, only for debug purposes")
     file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ui5)
     execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        $ENV{OPENUI5DIR} ${CMAKE_BINARY_DIR}/ui5/distribution)
  else()
    if(builtin_openui5)
      ROOT_CHECK_CONNECTION("builtin_openui5=OFF")
      if (NO_CONNECTION)
         message(SEND_ERROR "builtin_openui5=ON requires internet connection, check it or disable feature")
         list(APPEND HOTFIX_BUILD_FLAGS -Dbuiltin_openui5=OFF)
      endif()
      add_subdirectory(builtins/openui5)
    else()
      message(WARNING "Without builtin_openui5 option most of webgui components will not work")
    endif()
  endif()
  add_subdirectory(builtins/rendercore)
  add_subdirectory(builtins/mathjax)
endif()

#------------------------------------------------------------------------------------
# Check if we need libatomic to use atomic operations in the C++ code. On ARM systems
# we generally do. First just test if CMake is able to compile a test executable
# using atomic operations without the help of a library. Only if it can't do we start
# looking for libatomic for the build.
#
check_cxx_source_compiles("
#include <atomic>
#include <cstdint>
int main() {
   std::atomic<int> a1;
   int a1val = a1.load();
   (void)a1val;
   std::atomic<uint64_t> a2;
   uint64_t a2val = a2.load(std::memory_order_relaxed);
   (void)a2val;
   return 0;
}
" ROOT_HAVE_CXX_ATOMICS_WITHOUT_LIB)
set(ROOT_ATOMIC_LIBS)
if(NOT ROOT_HAVE_CXX_ATOMICS_WITHOUT_LIB)
  find_library(ROOT_ATOMIC_LIB NAMES atomic
    HINTS ENV LD_LIBRARY_PATH
    DOC "Path to the atomic library to use during the build")
  mark_as_advanced(ROOT_ATOMIC_LIB)
  if(ROOT_ATOMIC_LIB)
    set(ROOT_ATOMIC_LIBS ${ROOT_ATOMIC_LIB})
  endif()
endif()

#------------------------------------------------------------------------------------
# Check if we need to link -lstdc++fs to use <filesystem> (libstdc++ 8 and older).
set(_filesystem_source "
#include <filesystem>
int main(void) {
   std::filesystem::path p = \"path\";
   return 0;
}
")
check_cxx_source_compiles("${_filesystem_source}" ROOT_HAVE_NATIVE_CXX_FILESYSTEM)
if(NOT ROOT_HAVE_NATIVE_CXX_FILESYSTEM)
  set(CMAKE_REQUIRED_LIBRARIES stdc++fs)
  check_cxx_source_compiles("${_filesystem_source}" ROOT_NEED_STDCXXFS)
  if(NOT ROOT_NEED_STDCXXFS)
    message(FATAL_ERROR "Could not determine how to use C++17 <filesystem>")
  endif()
endif()

#------------------------------------------------------------------------------------
# Check if std::experimental::simd is available for vectorized TFormula and
# TMath features.
if(WIN32 OR APPLE OR CMAKE_CXX_STANDARD LESS 20)
  # Missing value means not available.
  set(ROOT_HAVE_EXPERIMENTAL_SIMD CACHE INTERNAL "Test ROOT_HAVE_EXPERIMENTAL_SIMD")
else()
  check_cxx_source_compiles("
      #include <experimental/simd>
      int main() {
          std::experimental::native_simd<int> v;
          return 0;
      }
  " ROOT_HAVE_EXPERIMENTAL_SIMD)
endif()

# On platforms with AVX-512, the libstdc++ implementation of
# <experimental/simd> (from GCC up to at least 16) fails to compile with
# non-GCC front ends (Clang, Intel icpx) because of a static_assert in the
# _VecBltnBtmsk (AVX-512 mask) ABI that requires `long long` and `long` to be
# the same type. The bug fires only for the AVX-512 mask ABI path, so we work
# around it by pinning ROOT's simd alias (Math/Types.h) to the 256-bit AVX2
# ABI variant instead of the platform-native ABI. That keeps the ABI of
# Float_v/Double_v/... consistent across all TUs (no `-mno-avx512f` needed)
# and lets the rest of ROOT keep its native AVX-512 codegen.
set(ROOT_EXPERIMENTAL_SIMD_PIN_AVX_ABI FALSE CACHE INTERNAL
    "Pin <experimental/simd> alias to the 256-bit ABI to dodge libstdc++ AVX-512 bug")
if(ROOT_HAVE_EXPERIMENTAL_SIMD)
  set(_simd_realistic_test "
      #include <experimental/simd>
      int main() {
          std::experimental::native_simd<double> a(1.0), b(2.0);
          where(a > b, a) = b;
          return 0;
      }
  ")
  check_cxx_source_compiles("${_simd_realistic_test}"
                            ROOT_EXPERIMENTAL_SIMD_FULL_USAGE_OK)
  if(NOT ROOT_EXPERIMENTAL_SIMD_FULL_USAGE_OK)
    set(_simd_pinned_abi_test "
        #include <experimental/simd>
        namespace stx = std::experimental;
        int main() {
            stx::simd<double, stx::simd_abi::__avx> a(1.0), b(2.0);
            where(a > b, a) = b;
            return 0;
        }
    ")
    check_cxx_source_compiles("${_simd_pinned_abi_test}"
                              ROOT_EXPERIMENTAL_SIMD_AVX_ABI_OK)
    if(ROOT_EXPERIMENTAL_SIMD_AVX_ABI_OK)
      set(ROOT_EXPERIMENTAL_SIMD_PIN_AVX_ABI TRUE CACHE INTERNAL "" FORCE)
      message(STATUS "Working around libstdc++ <experimental/simd> AVX-512 bug "
                     "by pinning Math/Types.h to the 256-bit AVX ABI")
    else()
      message(STATUS "Disabling experimental/simd-based features: libstdc++ "
                     "header fails to compile in this configuration")
      set(ROOT_HAVE_EXPERIMENTAL_SIMD FALSE CACHE INTERNAL "" FORCE)
    endif()
  endif()
endif()

#------------------------------------------------------------------------------------
# Check if the pyspark package is installed on the system.
# Needed to run tests of the distributed RDataFrame module that use pyspark.
# The functionality has been tested with pyspark 2.4 and above.
if(test_distrdf_pyspark)
  find_package(PySpark 2.4 REQUIRED)
endif()

#------------------------------------------------------------------------------------
# Check if the dask package is installed on the system.
# Needed to run tests of the distributed RDataFrame module that use dask.
if(test_distrdf_dask)
  find_package(Dask 2022.08.1 REQUIRED)
endif()
