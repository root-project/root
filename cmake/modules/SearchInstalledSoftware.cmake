#---Check for installed packages depending on the build options/components eamnbled -
include(ExternalProject)
include(FindPackageHandleStandardArgs)
set(repository_tarfiles http://service-spi.web.cern.ch/service-spi/external/tarFiles)

#---On MacOSX, try to find frameworks after standard libraries or headers------------
set(CMAKE_FIND_FRAMEWORK LAST)

#---Check for Cocoa/Quartz graphics backend (MacOS X only)
if(cocoa)
  if(APPLE)
    set(x11 OFF CACHE BOOL "" FORCE)
    set(builtin_freetype ON CACHE BOOL "" FORCE)
  else()
    message(STATUS "Cocoa option can only be enabled on MacOSX platform")
    set(cocoa OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Check for Zlib ------------------------------------------------------------------
if(NOT builtin_zlib)
  message(STATUS "Looking for ZLib")
  find_Package(ZLIB)
  if(NOT ZLIB_FOUND)
    message(STATUS "Zlib not found. Switching on builtin_zlib option")
    set(builtin_zlib ON CACHE BOOL "" FORCE)
   endif()
endif()
if(builtin_zlib)
  set(ZLIB_LIBRARY "" CACHE PATH "" FORCE)
endif()
if(ZLIB_LIBRARY)
  set(ZLIB_LIB ${ZLIB_LIBRARY})
endif()

#---Check for Freetype---------------------------------------------------------------
if(NOT builtin_freetype)
  message(STATUS "Looking for Freetype")
  find_package(Freetype)
  if(FREETYPE_FOUND)
    set(FREETYPE_INCLUDE_DIR ${FREETYPE_INCLUDE_DIR_freetype2})
  else()
    message(STATUS "FreeType not found. Switching on builtin_freetype option")
    set(builtin_freetype ON CACHE BOOL "" FORCE)
  endif()
endif()
if(builtin_freetype)  
  set(FREETYPE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/graf2d/freetype/freetype-2.3.12/include)
  set(FREETYPE_INCLUDE_DIRS ${FREETYPE_INCLUDE_DIR})
  if(WIN32)
    set(FREETYPE_LIBRARIES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/freetype.lib)
  else()
    set(FREETYPE_LIBRARIES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libfreetype.a)
  endif()
endif()

#---Check for PCRE-------------------------------------------------------------------
if(NOT builtin_pcre)
  message(STATUS "Looking for PCRE2")
  find_package(PCRE2)
  if(NOT PCRE2_FOUND)
    message(STATUS "PCRE2 not found. Looking for PCRE")
    find_package(PCRE)
    if(NOT PCRE_FOUND)
      message(STATUS "PCRE not found. Switching on builtin_pcre option")
      set(builtin_pcre ON CACHE BOOL "Enabled because PCRE not found (${builtin_pcre_description})" FORCE)
    endif()
  endif()
endif()
if(builtin_pcre)
  set(PCRE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/core/pcre/pcre-7.8)
  if(WIN32)
    set(PCRE_LIBRARIES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libpcre.lib) 
  else()
    set(PCRE_LIBRARIES "-L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lpcre") 
  endif()
endif()

#---Check for LZMA-------------------------------------------------------------------
if(NOT builtin_lzma)
  message(STATUS "Looking for LZMA")
  find_package(LZMA)
  if(LZMA_FOUND)
  else()
    message(STATUS "LZMA not found. Switching on builtin_lzma option")
    set(builtin_lzma ON CACHE BOOL "" FORCE)
  endif() 
endif()
if(builtin_lzma)
  set(lzma_version 5.0.3)
  message(STATUS "Building LZMA version ${lzma_version} included in ROOT itself")
  if(WIN32)
    ExternalProject_Add(
      LZMA
      URL ${CMAKE_SOURCE_DIR}/core/lzma/src/xz-${lzma_version}-win32.tar.gz 
      URL_MD5  65693dc257802b6778c28ed53ecca678
      PREFIX LZMA
      INSTALL_DIR ${CMAKE_BINARY_DIR}
       CONFIGURE_COMMAND "" BUILD_COMMAND ""
      INSTALL_COMMAND cmake -E copy lib/liblzma.dll <INSTALL_DIR>/bin
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      BUILD_IN_SOURCE 1
    )
    install(FILES ${CMAKE_BINARY_DIR}/LZMA/src/LZMA/lib/liblzma.dll DESTINATION ${CMAKE_INSTALL_BINDIR})
    set(LZMA_LIBRARIES ${CMAKE_BINARY_DIR}/LZMA/src/LZMA/lib/liblzma.lib)
    set(LZMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/LZMA/src/LZMA/include)
  else() 
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
      set(LZMA_CFLAGS "-Wno-format-nonliteral")
      set(LZMA_LDFLAGS "-Qunused-arguments")
    elseif( CMAKE_CXX_COMPILER_ID STREQUAL Intel)
      set(LZMA_CFLAGS "-wd188 -wd181 -wd1292 -wd10006 -wd10156 -wd2259 -wd981 -wd128 -wd3179 -wd2102")
    endif()
    ExternalProject_Add(
      LZMA
      URL ${CMAKE_SOURCE_DIR}/core/lzma/src/xz-${lzma_version}.tar.gz 
      URL_MD5 858405e79590e9b05634c399497f4ba7
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR> --libdir <INSTALL_DIR>/lib
                        --with-pic --disable-shared --quiet
                        CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${LZMA_CFLAGS} LDFLAGS=${LZMA_LDFLAGS}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      BUILD_IN_SOURCE 1
     )
    set(LZMA_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lzma${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(LZMA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
  endif()
endif()


#---Check for LZ4--------------------------------------------------------------------
if(NOT builtin_lz4)
  message(STATUS "Looking for LZ4")
  find_package(LZ4)
  if(LZ4_FOUND)
  else()
    message(STATUS "LZ4 not found. Switching on builtin_lz4 option")
    set(builtin_lz4 ON CACHE BOOL "" FORCE)
  endif()
endif()
# Note: the above if-statement may change the value of builtin_lz4 to ON.
if(builtin_lz4)
  set(lz4_version v1.7.5)
  message(STATUS "Building LZ4 version ${lz4_version} included in ROOT itself")
  if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    set(LZ4_CFLAGS "-Wno-format-nonliteral")
  elseif( CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    set(LZ4_CFLAGS "-wd188 -wd181 -wd1292 -wd10006 -wd10156 -wd2259 -wd981 -wd128 -wd3179")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(LZ4_CFLAGS "/Zl")
  endif()
  set(LZ4_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lz4${CMAKE_STATIC_LIBRARY_SUFFIX})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/include" NATIVE_INCLUDEDIR)
    ExternalProject_Add(
      LZ4
      URL http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/lz4-${lz4_version}.tar.gz
      URL_MD5 c9610c5ce97eb431dddddf0073d919b9
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND cl /c "${LZ4_CFLAGS}" -DXXH_NAMESPACE=LZ4_ lib/lz4.c lib/lz4hc.c lib/lz4frame.c lib/xxhash.c
      BUILD_COMMAND lib /NODEFAULTLIB lz4.obj lz4hc.obj lz4frame.obj xxhash.obj /OUT:${LZ4_LIBRARIES}
      INSTALL_COMMAND xcopy "lib\\*.h" "${NATIVE_INCLUDEDIR}\\" /Y 
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
    )
  else()
    ExternalProject_Add(
      LZ4
      URL http://lcgpackages.web.cern.ch/lcgpackages/tarFiles/sources/lz4-${lz4_version}.tar.gz
      URL_MD5 c9610c5ce97eb431dddddf0073d919b9
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND  /bin/sh -c "PREFIX=<INSTALL_DIR> make cmake"
      BUILD_COMMAND /bin/sh -c "PREFIX=<INSTALL_DIR> MOREFLAGS=-fPIC make"
      INSTALL_COMMAND /bin/sh -c "PREFIX=<INSTALL_DIR> make install"
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1 BUILD_IN_SOURCE 1
    )
  endif()
  set(LZ4_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
  set(LZ4_DEFINITIONS -DBUILTIN_LZ4)
endif()


#---Check for X11 which is mandatory lib on Unix--------------------------------------
if(x11)
  message(STATUS "Looking for X11")
  if(X11_X11_INCLUDE_PATH)
    set(X11_FIND_QUIETLY 1)
  endif()
  find_package(X11 REQUIRED)
  if(X11_FOUND)
    list(REMOVE_DUPLICATES X11_INCLUDE_DIR)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_INCLUDE_DIR: ${X11_INCLUDE_DIR}")
      message(STATUS "X11_LIBRARIES: ${X11_LIBRARIES}")
    endif()
  else()
    message(FATAL_ERROR "libX11 and X11 headers must be installed.")
  endif()
  if(X11_Xpm_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xpm_INCLUDE_PATH: ${X11_Xpm_INCLUDE_PATH}")
      message(STATUS "X11_Xpm_LIB: ${X11_Xpm_LIB}")
    endif()
  else()
    message(FATAL_ERROR "libXpm and Xpm headers must be installed.")
  endif()
  if(X11_Xft_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xft_INCLUDE_PATH: ${X11_Xft_INCLUDE_PATH}")
      message(STATUS "X11_Xft_LIB: ${X11_Xft_LIB}")
    endif()
    set(xft ON)
  else()
    message(FATAL_ERROR "libXft and Xft headers must be installed.")
  endif()
  if(X11_Xext_FOUND)
    if(NOT X11_FIND_QUIETLY)
      message(STATUS "X11_Xext_INCLUDE_PATH: ${X11_Xext_INCLUDE_PATH}")
      message(STATUS "X11_Xext_LIB: ${X11_Xext_LIB}")
    endif()
  else()
    message(FATAL_ERROR "libXext and Xext headers must be installed.")
  endif()
else()
  set(xft OFF)
endif()


#---Check for AfterImage---------------------------------------------------------------
if(NOT builtin_afterimage)
  message(STATUS "Looking for AfterImage")
  find_package(AfterImage)
  if(NOT AFTERIMAGE_FOUND)
    message(STATUS "AfterImage not found. Switching on builtin_afterimage option")
    set(builtin_afterimage ON CACHE BOOL "" FORCE)    
  endif()
endif()

#---Check for all kind of graphics includes needed by libAfterImage--------------------
if(asimage)
  if(NOT x11 AND NOT cocoa AND NOT WIN32)
    message(STATUS "Switching off 'asimage' because neither 'x11' nor 'cocoa' nor 'WIN32' are enabled")
    set(asimage OFF CACHE BOOL "" FORCE)
  endif()
endif()
if(asimage)
  set(ASEXTRA_LIBRARIES)
  find_Package(GIF)
  if(GIF_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${GIF_LIBRARIES})
  endif()
  find_Package(TIFF)
  if(TIFF_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${TIFF_LIBRARIES})
  endif()
  find_Package(PNG)
  if(PNG_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${PNG_LIBRARIES})
    list(GET PNG_INCLUDE_DIRS 0 PNG_INCLUDE_DIR)
  endif()
  find_Package(JPEG)
  if(JPEG_FOUND)
    set(ASEXTRA_LIBRARIES ${ASEXTRA_LIBRARIES} ${JPEG_LIBRARIES})
  endif()
endif()

#---Check for GSL library---------------------------------------------------------------
if(mathmore OR builtin_gsl)
  message(STATUS "Looking for GSL")
  if(NOT builtin_gsl)
    find_package(GSL 1.10)
    if(NOT GSL_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "GSL package not found and 'mathmore' component if required ('fail-on-missing' enabled). "
                            "Alternatively, you can enable the option 'builtin_gsl' to build the GSL libraries internally.")
      else()
        message(STATUS "GSL not found. Set variable GSL_DIR to point to your GSL installation")
        message(STATUS "               Alternatively, you can also enable the option 'builtin_gsl' to build the GSL libraries internally'")
        message(STATUS "               For the time being switching OFF 'mathmore' option")
        set(mathmore OFF CACHE BOOL "" FORCE)
      endif()
    endif()
  else()
    set(gsl_version 1.15)
    message(STATUS "Downloading and building GSL version ${gsl_version}") 
    ExternalProject_Add(
      GSL
      # http://mirror.switch.ch/ftp/mirror/gnu/gsl/gsl-${gsl_version}.tar.gz
      URL ${repository_tarfiles}/gsl-${gsl_version}.tar.gz
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR> --enable-shared=no CFLAGS=${CMAKE_C_FLAGS}
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
    )
    set(GSL_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
    foreach(l gsl gslcblas)
      list(APPEND GSL_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${l}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach()
    set(mathmore ON CACHE BOOL "" FORCE)
  endif()
endif()


#---Check for Python installation-------------------------------------------------------
if(python)
  message(STATUS "Looking for Python")
  #---First look for the python interpreter and fix the version of it for the libraries--
  find_package(PythonInterp)
  if(PYTHONINTERP_FOUND)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"
                    OUTPUT_VARIABLE PYTHON_VERSION)
    message(STATUS "Found Python interpreter version ${PYTHON_VERSION}")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(sys.prefix)"
                    OUTPUT_VARIABLE PYTHON_PREFIX)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_PREFIX})
  endif()
  find_package(PythonLibs)
  if(NOT PYTHONLIBS_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "PythonLibs package not found and python component required")
    else()
      set(python OFF CACHE BOOL "" FORCE)
      message(STATUS "Python not found. Switching off python option")
    endif()
  else()
  endif()
endif()

#---Check for Ruby installation-------------------------------------------------------
if(ruby)
  message(STATUS "Looking for Ruby")
  find_package(Ruby)
  if(NOT RUBY_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Ruby package not found and ruby component required")
    else()
      set(ruby OFF CACHE BOOL "" FORCE)
      message(STATUS "Ruby not found. Switching off ruby option")
    endif()
  else()
    string(REGEX REPLACE "([0-9]+).*$" "\\1" RUBY_MAJOR_VERSION "${RUBY_VERSION}")
    string(REGEX REPLACE "[0-9]+\\.([0-9]+).*$" "\\1" RUBY_MINOR_VERSION "${RUBY_VERSION}")
    string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" RUBY_PATCH_VERSION "${RUBY_VERSION}")
  endif()
endif()

#---Check for GCCXML installation-------------------------------------------------------
if(cintex OR reflex)
  message(STATUS "Looking for GCCXML")
  find_package(GCCXML)
  if(GCCXML_FOUND)
    set(gccxml ${GCCXML_EXECUTABLE})
  else()
    if(fail-on-missing)
      message(STATUS "GCCXML not found and cintex or reflex option required. Continuing")
    endif()    
  endif()
endif()

#---Check for OpenGL installation-------------------------------------------------------
if(opengl)
  message(STATUS "Looking for OpenGL")
  if(APPLE AND NOT cocoa)
    find_path(OPENGL_INCLUDE_DIR GL/gl.h  PATHS /usr/X11R6/include)
    find_library(OPENGL_gl_LIBRARY NAMES GL PATHS /usr/X11R6/lib)
    find_library(OPENGL_glu_LIBRARY NAMES GLU PATHS /usr/X11R6/lib)
    find_package_handle_standard_args(OpenGL REQUIRED_VARS OPENGL_INCLUDE_DIR OPENGL_gl_LIBRARY OPENGL_glu_LIBRARY)
    set(OPENGL_LIBRARIES ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
    mark_as_advanced(OPENGL_INCLUDE_DIR OPENGL_glu_LIBRARY OPENGL_gl_LIBRARY)
  else()
    find_package(OpenGL)
  endif()
  if(NOT OPENGL_FOUND OR NOT OPENGL_GLU_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "OpenGL package (with GLU) not found and opengl option required")
    else()
      message(STATUS "OpenGL (with GLU) not found. Switching off opengl option")
      set(opengl OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Graphviz installation-------------------------------------------------------
if(gviz)
  message(STATUS "Looking for Graphviz")
  find_package(Graphviz)
  if(NOT GRAPHVIZ_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Graphviz package not found and gviz option required")
    else()
      message(STATUS "Graphviz not found. Switching off gviz option")
      set(gviz OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Qt installation-------------------------------------------------------
if(qt OR qtgsi)
  message(STATUS "Looking for Qt4")
  find_package(Qt4 COMPONENTS QtCore QtGui)
  if(NOT QT4_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Qt4 package not found and qt/qtgsi component required")
    else()
      message(STATUS "Qt4 not found. Switching off qt/qtgsi option")
      set(qt OFF CACHE BOOL "" FORCE)
      set(qtgsi OFF CACHE BOOL "" FORCE)
    endif()
  else()
    MATH(EXPR QT_VERSION_NUM "${QT_VERSION_MAJOR}*10000 + ${QT_VERSION_MINOR}*100 + ${QT_VERSION_PATCH}")
  endif()
endif()


#---Check for Bonjour installation-------------------------------------------------------
if(bonjour)
  message(STATUS "Looking for Bonjour")
  find_package(Bonjour)
  if(NOT BONJOUR_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Bonjour/Avahi libraries not found and Bonjour component required")
    else()
      message(STATUS "Bonjour not found. Switching off bonjour option")
      set(bonjour OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()


#---Check for krb5 Support-----------------------------------------------------------
if(krb5)
  message(STATUS "Looking for Kerberos 5")
  find_package(Kerberos5)
  if(NOT KRB5_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Kerberos 5 libraries not found and they are required")
    else()
      message(STATUS "Kerberos 5 not found. Switching off krb5 option")
      set(krb5 OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

if(krb5 OR afs)
  find_library(COMERR_LIBRARY com_err)
  if(COMERR_LIBRARY)
    set(COMERR_LIBRARIES ${COMERR_LIBRARY})
  endif()
endif()

#---Check for XML Parser Support-----------------------------------------------------------
if(xml)
  message(STATUS "Looking for LibXml2")
  find_package(LibXml2)
  if(NOT LIBXML2_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "LibXml2 libraries not found and they are required (xml option enabled)")
    else()
      message(STATUS "LibXml2 not found. Switching off xml option")
      set(xml OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for OpenSSL------------------------------------------------------------------
if(ssl OR builtin_openssl)
  if(builtin_openssl)
    set(openssl_version 1.0.2d)
    message(STATUS "Downloading and building OpenSSL version ${openssl_version}")
    if(APPLE)
      set(openssl_config_cmd ./Configure darwin64-x86_64-cc)
    else()
      set(openssl_config_cmd ./config)
    endif()
    ExternalProject_Add(
      OPENSSL
      URL ${repository_tarfiles}/openssl-${openssl_version}.tar.gz
      CONFIGURE_COMMAND ${openssl_config_cmd} no-shared --prefix=<INSTALL_DIR>
      BUILD_COMMAND make -j1 CC=${CMAKE_C_COMPILER}\ -fPIC
      INSTALL_COMMAND make install_sw
      BUILD_IN_SOURCE 1
      LOG_BUILD 1 LOG_CONFIGURE 1 LOG_DOWNLOAD 1 LOG_INSTALL 1
    )
    ExternalProject_Get_Property(OPENSSL INSTALL_DIR)
    set(OPENSSL_INCLUDE_DIR ${INSTALL_DIR}/include)
    set(OPENSSL_LIBRARIES ${INSTALL_DIR}/lib/libssl.a ${INSTALL_DIR}/lib/libcrypto.a)
    set(OPENSSL_PREFIX ${INSTALL_DIR})
    set(ssl ON CACHE BOOL "" FORCE)
  else()
    message(STATUS "Looking for OpenSSL")
    find_package(OpenSSL)
    if(NOT OPENSSL_FOUND)
      if(fail-on-missing)
        message(FATAL_ERROR "OpenSSL libraries not found and they are required (ssl option enabled)")
      else()
        message(STATUS "OpenSSL not found. Switching off ssl option")
        set(ssl OFF CACHE BOOL "" FORCE)
      endif()
    endif()
  endif()
endif()

#---Check for Castor-------------------------------------------------------------------
if(castor OR rfio)
  message(STATUS "Looking for Castor")
  find_package(Castor)
  if(NOT CASTOR_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Castor libraries not found and they are required (castor option enabled)")
    else()
      message(STATUS "Castor not found. Switching off castor/rfio option")
      set(castor OFF CACHE BOOL "" FORCE)
      set(rfio OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for MySQL-------------------------------------------------------------------
if(mysql)
  message(STATUS "Looking for MySQL")
  find_package(MySQL)
  if(NOT MYSQL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "MySQL libraries not found and they are required (mysql option enabled)")
    else()
      message(STATUS "MySQL not found. Switching off mysql option")
      set(mysql OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Oracle-------------------------------------------------------------------
if(oracle)
  message(STATUS "Looking for Oracle")
  find_package(Oracle)
  if(NOT ORACLE_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Oracle libraries not found and they are required (orable option enabled)")
    else()
      message(STATUS "Oracle not found. Switching off oracle option")
      set(oracle OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for ODBC-------------------------------------------------------------------
if(odbc)
  message(STATUS "Looking for ODBC")
  find_package(ODBC)
  if(NOT ODBC_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "ODBC libraries not found and they are required (odbc option enabled)")
    else()
      message(STATUS "ODBC not found. Switching off odbc option")
      set(odbc OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for PostgreSQL-------------------------------------------------------------------
if(pgsql)
  message(STATUS "Looking for PostgreSQL")
  find_package(PostgreSQL)
  if(NOT POSTGRESQL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "PostgreSQL libraries not found and they are required (pgsql option enabled)")
    else()
      message(STATUS "PostgreSQL not found. Switching off pgsql option")
      set(pgsql OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for SQLite-------------------------------------------------------------------
if(sqlite)
  message(STATUS "Looking for SQLite")
  find_package(Sqlite)
  if(NOT SQLITE_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "SQLite libraries not found and they are required (sqlite option enabled)")
    else()
      message(STATUS "SQLite not found. Switching off sqlite option")
      set(sqlite OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Pythia6-------------------------------------------------------------------
if(pythia6)
  message(STATUS "Looking for Pythia6")
  find_package(Pythia6 QUIET)
  if(NOT PYTHIA6_FOUND AND NOT pythia6_nolink)
    if(fail-on-missing)
      message(FATAL_ERROR "Pythia6 libraries not found and they are required (pythia6 option enabled)")
    else()
      message(STATUS "Pythia6 not found. Switching off pythia6 option")
      set(pythia6 OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Pythia8-------------------------------------------------------------------
if(pythia8)
  message(STATUS "Looking for Pythia8")
  find_package(Pythia8)
  if(NOT PYTHIA8_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Pythia8 libraries not found and they are required (pythia8 option enabled)")
    else()
      message(STATUS "Pythia8 not found. Switching off pythia8 option")
      set(pythia8 OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for FFTW3-------------------------------------------------------------------
if(fftw3)
  message(STATUS "Looking for FFTW3")
  find_package(FFTW)
  if(NOT FFTW_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "FFTW3 libraries not found and they are required (fftw3 option enabled)")
    else()
      message(STATUS "FFTW3 not found. Switching off fftw3 option")
      set(fftw3 OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for fitsio-------------------------------------------------------------------
if(fitsio OR builtin_cfitsio)
  if(builtin_cfitsio)
    set(cfitsio_version 3.280)
    string(REPLACE "." "" cfitsio_version_no_dots ${cfitsio_version})
    message(STATUS "Downloading and building CFITSIO version ${cfitsio_version}") 
    ExternalProject_Add(
      CFITSIO
      # ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio${cfitsio_version_no_dots}.tar.gz
      URL ${repository_tarfiles}/cfitsio${cfitsio_version_no_dots}.tar.gz
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix <INSTALL_DIR>
      LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
      BUILD_IN_SOURCE 1
    )
    set(CFITSIO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
    set(CFITSIO_LIBRARIES ${CMAKE_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}cfitsio${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(fitsio ON CACHE BOOL "" FORCE)
  else()
    message(STATUS "Looking for CFITSIO")  
    find_package(CFITSIO)
    if(NOT CFITSIO_FOUND)
      message(STATUS "CFITSIO not found. You can enable the option 'builtin_cfitsio' to build the library internally'") 
      message(STATUS "                   For the time being switching off 'fitsio' option")
      set(fitsio OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()


#---Check Shadow password support----------------------------------------------------
if(shadowpw)
  if(NOT EXISTS /etc/shadow)  #---TODO--The test always succeeds because the actual file is protected
    if(NOT CMAKE_SYSTEM_NAME MATCHES Linux)
      message(STATUS "Support Shadow password not found. Switching off shadowpw option")
      set(shadowpw OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Alien support----------------------------------------------------------------
if(alien)
  find_package(Alien)
  if(NOT ALIEN_FOUND)
    message(STATUS "Alien API not found. Set variable ALIEN_DIR to point to your Alien installation")
    message(STATUS "For the time being switching OFF 'alien' option")
    set(alien OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Monalisa support----------------------------------------------------------------
if(monalisa)
  find_package(Monalisa)
  if(NOT MONALISA_FOUND)
    message(STATUS "Monalisa not found. Set variable MONALISA_DIR to point to your Monalisa installation")
    message(STATUS "For the time being switching OFF 'monalisa' option")
    set(monalisa OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Check for Xrootd support---------------------------------------------------------
if(xrootd)
  message(STATUS "Looking for XROOTD")
  if(NOT builtin_xrootd)
    find_package(XROOTD)
    if(NOT XROOTD_FOUND)
      message(STATUS "XROOTD not found. Set environment variable XRDSYS to point to your XROOTD installation")
      message(STATUS "                  Alternatively, you can also enable the option 'builtin_xrootd' to build XROOTD  internally'")
      message(STATUS "                  For the time being switching OFF 'xrootd' option")
      set(xrootd OFF CACHE BOOL "" FORCE)
    else()
      set(xrootd_versionnum ${xrdversnum})  # variable used internally
    endif()
  endif()
endif()
if(builtin_xrootd)
  set(xrootd_version 4.2.2)
  set(xrootd_versionnum 400020002)
  message(STATUS "Downloading and building XROOTD version ${xrootd_version}")
  string(REPLACE "-Wall " "" __cxxflags "${CMAKE_CXX_FLAGS}")  # Otherwise it produces many warnings
  string(REPLACE "-W " "" __cxxflags "${__cxxflags}")          # Otherwise it produces many warnings
  ExternalProject_Add(
    XROOTD
    URL http://xrootd.org/download/v${xrootd_version}/xrootd-${xrootd_version}.tar.gz
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_CXX_FLAGS=${__cxxflags}
               -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
               -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
               -DENABLE_PYTHON=OFF
    LOG_DOWNLOAD 1 LOG_CONFIGURE 1 LOG_BUILD 1 LOG_INSTALL 1
  )
  # We cannot call find_package(XROOTD) becuase the package is not yet built. So, we need to emulate what it defines....
  set(_LIBDIR_DEFAULT "lib")
  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND NOT CMAKE_CROSSCOMPILING AND NOT EXISTS "/etc/debian_version")
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
      set(_LIBDIR_DEFAULT "lib64")
    endif()
  endif()
  set(XROOTD_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include/xrootd ${CMAKE_BINARY_DIR}/include/xrootd/private)
  set(XROOTD_LIBRARIES ${CMAKE_BINARY_DIR}/${_LIBDIR_DEFAULT}/libXrdUtils${CMAKE_SHARED_LIBRARY_SUFFIX}
                       ${CMAKE_BINARY_DIR}/${_LIBDIR_DEFAULT}/libXrdClient${CMAKE_SHARED_LIBRARY_SUFFIX}
                       ${CMAKE_BINARY_DIR}/${_LIBDIR_DEFAULT}/libXrdCl${CMAKE_SHARED_LIBRARY_SUFFIX})
  if(xrootd_version VERSION_LESS 4)
    list(APPEND XROOTD_LIBRARIES ${CMAKE_BINARY_DIR}/${_LIBDIR_DEFAULT}/libXrdMain${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(XROOTD_NOMAIN TRUE)
  endif()
  set(XROOTD_CFLAGS "-DROOTXRDVERS=${xrootd_versionnum}")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/${_LIBDIR_DEFAULT}/ DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    COMPONENT libraries
                    FILES_MATCHING PATTERN "libXrd*")
  set(xrootd ON CACHE BOOL "" FORCE)
endif()
if(xrootd AND xrootd_versionnum VERSION_GREATER 300030005)
  set(netxng ON)
else()
  set(netxng OFF)
endif()

#---Check for gfal-------------------------------------------------------------------
if(gfal)
  find_package(GFAL)
  if(NOT GFAL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "Gfal library not found and is required (gfal option enabled)")
    else()
      message(STATUS "GFAL library not found. Set variable GFAL_DIR to point to your gfal installation
                      and the variable SRM_IFCE_DIR to the srm_ifce installation")
      message(STATUS "For the time being switching OFF 'gfal' option")
      set(gfal OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()


#---Check for dCache-------------------------------------------------------------------
if(dcache)
  find_package(DCAP)
  if(NOT DCAP_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "dCap library not found and is required (dcache option enabled)")
    else()
      message(STATUS "dCap library not found. Set variable DCAP_DIR to point to your dCache installation")
      message(STATUS "For the time being switching OFF 'dcache' option")
      set(dcache OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for Ldap--------------------------------------------------------------------
if(ldap)
  find_package(Ldap)
  if(NOT LDAP_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "ldap library not found and is required (ldap option enabled)")
    else()
      message(STATUS "ldap library not found. Set variable LDAP_DIR to point to your ldap installation")
      message(STATUS "For the time being switching OFF 'ldap' option")
      set(ldap OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for globus--------------------------------------------------------------------
if(globus)
  find_package(Globus)
  if(NOT GLOBUS_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "globus libraries not found and is required ('globus' option enabled)")
    else()
      message(STATUS "globus libraries not found. Set environment var GLOBUS_LOCATION or varibale GLOBUS_DIR to point to your globus installation")
      message(STATUS "For the time being switching OFF 'globus' option")
      set(globus OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for ftgl if needed----------------------------------------------------------
if(NOT builtin_ftgl)
  find_package(FTGL)
  if(NOT FTGL_FOUND)
    if(fail-on-missing)
      message(FATAL_ERROR "ftgl library not found and is required ('builtin_ftgl' is OFF). Set varible FTGL_ROOT_DIR to installation location")
    else()
      message(STATUS "ftgl library not found. Set variable FTGL_ROOT_DIR to point to your installation")
      message(STATUS "For the time being switching ON 'builtin_ftgl' option")
      set(builtin_ftgl ON CACHE BOOL "" FORCE)
    endif()
  endif()
endif()
if(builtin_ftgl)
  set(FTGL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/graf3d/ftgl/inc)
  set(FTGL_CFLAGS -DBUILTIN_FTGL)
  set(FTGL_LIBRARIES FTGL)
endif()

#---Check for DavIx library-----------------------------------------------------------
if(davix OR builtin_davix)
  if(builtin_davix)
    if(NOT davix)
      set(davix ON CACHE BOOL "" FORCE)
    endif()
    set(DAVIX_VERSION 0.3.6)
    message(STATUS "Downloading and building Davix version ${DAVIX_VERSION}")
    string(REPLACE "-Wall " "" __cxxflags "${CMAKE_CXX_FLAGS}")                      # Otherwise it produces tones of warnings
    string(REPLACE "-W " "" __cxxflags "${__cxxflags}")
    string(REPLACE "-Wall " "" __cflags "${CMAKE_C_FLAGS}")                          # Otherwise it produces tones of warnings
    string(REPLACE "-W " "" __cflags "${__cflags}")
    ROOT_ADD_CXX_FLAG(__cxxflags -Wno-unused-const-variable)
    ROOT_ADD_C_FLAG(__cflags -Wno-format)
    ROOT_ADD_C_FLAG(__cflags -Wno-implicit-function-declaration)
    ExternalProject_Add(
      DAVIX
      # http://grid-deployment.web.cern.ch/grid-deployment/dms/lcgutil/tar/davix/davix-embedded-${DAVIX_VERSION}.tar.gz
      URL ${repository_tarfiles}/davix-embedded-${DAVIX_VERSION}.tar.gz
      # Patch need. see https://github.com/cern-it-sdc-id/davix/issues/6
      PATCH_COMMAND patch -p1 -i ${CMAKE_SOURCE_DIR}/cmake/patches/davix-${DAVIX_VERSION}.patch 
      CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:STRING=${OPENSSL_PREFIX}
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} 
                 -DBOOST_EXTERNAL=OFF
                 -DSTATIC_LIBRARY=ON
                 -DSHARED_LIBRARY=OFF
                 -DENABLE_TOOLS=OFF
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_C_FLAGS=${__cflags}
                 -DCMAKE_CXX_FLAGS=${__cxxflags}
                 -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
                 -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
      LOG_BUILD 1 LOG_CONFIGURE 1 LOG_DOWNLOAD 1 LOG_INSTALL 1
    )
    ExternalProject_Get_Property(DAVIX INSTALL_DIR)
    if(${SYSCTL_OUTPUT} MATCHES x86_64)
      set(_LIBDIR "lib64")
    else()
      set(_LIBDIR "lib")
    endif()
    set(DAVIX_INCLUDE_DIR ${INSTALL_DIR}/include/davix)
    set(DAVIX_LIBRARY ${INSTALL_DIR}/${_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}davix${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(DAVIX_INCLUDE_DIRS ${DAVIX_INCLUDE_DIR})
    foreach(l davix neon boost_static_internal)
      list(APPEND DAVIX_LIBRARIES ${INSTALL_DIR}/${_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${l}${CMAKE_STATIC_LIBRARY_SUFFIX})
    endforeach()
    if(builtin_openssl)
      add_dependencies(DAVIX OPENSSL)  # Build first OpenSSL
    endif()
  else()
    message(STATUS "Looking for DAVIX")
    find_package(Davix)
    if(NOT DAVIX_FOUND)
      message(STATUS "Davix not found. You can enable the option 'builtin_davix' to build the library internally'")
      message(STATUS "                 For the time being switching off 'davix' option")
      set(davix OFF CACHE BOOL "" FORCE)
    endif()
  endif()
endif()

#---Check for vc and its compatibility-----------------------------------------------
if(vc)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.5)
      message(STATUS "VC requires GCC version >= 4.5; switching OFF 'vc' option")
      set(vc OFF CACHE BOOL "" FORCE)
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
      message(STATUS "VC requires Clang version >= 4.0; switching OFF 'vc' option")
      set(vc OFF CACHE BOOL "" FORCE)
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "VC is not supported on Windows; switching OFF 'vc' option")
    set(vc OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Report non implemented options---------------------------------------------------
foreach(opt afs chirp glite hdfs pch sapdb srp)
  if(${opt})
    message(STATUS ">>> Option '${opt}' not implemented yet! Signal your urgency to pere.mato@cern.ch")
    set(${opt} OFF CACHE BOOL "" FORCE)
  endif()
endforeach()

