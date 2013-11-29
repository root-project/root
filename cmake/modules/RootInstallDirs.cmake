# - Define GNU standard installation directories
# Provides install directory variables as defined for GNU software:
#  http://www.gnu.org/prep/standards/html_node/Directory-Variables.html
# Inclusion of this module defines the following variables:
#  CMAKE_INSTALL_<dir>      - destination for files of a given type
#  CMAKE_INSTALL_FULL_<dir> - corresponding absolute path
# where <dir> is one of:
#  BINDIR           - user executables (bin)
#  LIBDIR           - object code libraries (lib or lib64 or lib/<multiarch-tuple> on Debian)
#  INCDIR           - C/C++ header files (include)
#  ETCDIR           - read-only single-machine data (etc)
#  DATAROOTDIR      - read-only architecture-independent data (share/root)
#  DATADIR          - read-only architecture-independent data (DATAROOTDIR)
#  MANDIR           - man documentation (DATAROOTDIR/man)
#  MACRODIR         - ROOT macros (DATAROOTDIR/macros)
#  CINTINCDIR       - CINT include files (LIBDIR/macros)
#  ICONDIR          - icons (DATAROOTDIR/icons)
#  SRCDIR           - sources (DATAROOTDIR/src)
#  FONTDIR          - fonts (DATAROOTDIR/fonts)
#  DOCDIR           - documentation root (DATAROOTDIR/doc/PROJECT_NAME)
#  TESTDIR          - tests (DOCDIR/test)
#  TUTDIR           - tutorials (DOCDIR/tutorials)
#  ACLOCALDIR       - locale-dependent data (DATAROOTDIR/aclocal)
#  CMAKEDIR         - cmake modules (DATAROOTDIR/cmake)
#  ELISPDIR         - lisp files (DATAROOTDIR/emacs/site-lisp)
#
# Each CMAKE_INSTALL_<dir> value may be passed to the DESTINATION options of
# install() commands for the corresponding file type.  If the includer does
# not define a value the above-shown default will be used and the value will
# appear in the cache for editing by the user.
# Each CMAKE_INSTALL_FULL_<dir> value contains an absolute path constructed
# from the corresponding destination by prepending (if necessary) the value
# of CMAKE_INSTALL_PREFIX.

#=============================================================================
# Copyright 2011 Nikita Krupen'ko <krnekit@gmail.com>
# Copyright 2011 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

if(NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
  if(gnuinstall)
    set(CMAKE_INSTALL_LIBDIR "lib/root" CACHE PATH "object code libraries (lib/root)")
  else()
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "object code libraries (lib)")
  endif()
endif()

if(NOT DEFINED CMAKE_INSTALL_CINTINCDIR)
  if(gnuinstall)
    set(CMAKE_INSTALL_CINTINCDIR "lib/root/cint" CACHE PATH "cint includes and libraries libraries (cint)")
  else()
    set(CMAKE_INSTALL_CINTINCDIR "cint" CACHE PATH "cint includes and libraries libraries (cint)")
  endif()
endif()

if(NOT DEFINED CMAKE_INSTALL_INCDIR)
  if(gnuinstall)
    set(CMAKE_INSTALL_INCDIR "include/root" CACHE PATH "C header files (include)")
  else()
    set(CMAKE_INSTALL_INCDIR "include" CACHE PATH "C header files (include)")
  endif()
endif()

if(NOT DEFINED CMAKE_INSTALL_ETCDIR)
  if(gnuinstall)
    set(CMAKE_INSTALL_ETCDIR "/etc/root" CACHE PATH "read-only single-machine data (etc)")
  else()
    set(CMAKE_INSTALL_ETCDIR "etc" CACHE PATH "read-only single-machine data (etc)")
  endif()
endif()

if(gnuinstall)
  set(__dataroot__ share/root/)
  set(__share__ share/)
endif()

#-----------------------------------------------------------------------------
# Values whose defaults are relative to DATAROOTDIR.  Store empty values in
# the cache and store the defaults in local variables if the cache values are
# not set explicitly.  This auto-updates the defaults as DATAROOTDIR changes.

if(NOT CMAKE_INSTALL_DATADIR)
  set(CMAKE_INSTALL_DATADIR "" CACHE PATH "read-only architecture-independent data (DATAROOTDIR)")
  if(gnuinstall)
    set(CMAKE_INSTALL_DATADIR "${__share__}root")
  else()
    set(CMAKE_INSTALL_DATADIR ".")
  endif()
endif()

if(NOT CMAKE_INSTALL_MANDIR)
  set(CMAKE_INSTALL_MANDIR "" CACHE PATH "man documentation (DATAROOTDIR/man)")
  set(CMAKE_INSTALL_MANDIR "${__share__}man")
endif()

if(NOT CMAKE_INSTALL_MACRODIR)
  set(CMAKE_INSTALL_MACRODIR "" CACHE PATH "macros documentation (DATAROOTDIR/macros)")
  set(CMAKE_INSTALL_MACRODIR "${__dataroot__}macros")
endif()

if(NOT CMAKE_INSTALL_ICONDIR)
  set(CMAKE_INSTALL_ICONDIR "" CACHE PATH "icons (DATAROOTDIR/icons)")
  set(CMAKE_INSTALL_ICONDIR "${__dataroot__}icons")
endif()

if(NOT CMAKE_INSTALL_FONTDIR)
  set(CMAKE_INSTALL_FONTDIR "" CACHE PATH "fonts (DATAROOTDIR/fonts)")
  set(CMAKE_INSTALL_FONTDIR "${__dataroot__}fonts")
endif()

if(NOT CMAKE_INSTALL_SRCDIR)
  set(CMAKE_INSTALL_SRCDIR "" CACHE PATH "sources (DATAROOTDIR/src)")
  set(CMAKE_INSTALL_SRCDIR "${__dataroot__}src")
endif()

if(NOT CMAKE_INSTALL_ACLOCALDIR)
  set(CMAKE_INSTALL_ACLOCALDIR "" CACHE PATH "locale-dependent data (DATAROOTDIR/aclocal)")
  set(CMAKE_INSTALL_ACLOCALDIR "${__share__}aclocal")
endif()

if(NOT CMAKE_INSTALL_CMAKEDIR)
  set(CMAKE_INSTALL_CMAKEDIR "" CACHE PATH "CMake modules (DATAROOTDIR/cmake)")
  set(CMAKE_INSTALL_CMAKEDIR "${__dataroot__}cmake")
endif()

if(NOT CMAKE_INSTALL_ELISPDIR)
  set(CMAKE_INSTALL_ELISPDIR "" CACHE PATH "Lisp files (DATAROOTDIR/emacs/site-lisp)")
  set(CMAKE_INSTALL_ELISPDIR "${__share__}emacs/site-lisp")
endif()

if(NOT CMAKE_INSTALL_DOCDIR)
  set(CMAKE_INSTALL_DOCDIR "" CACHE PATH "documentation root (DATAROOTDIR/doc/PROJECT_NAME)")
  if(gnuinstall)
    set(CMAKE_INSTALL_DOCDIR "${__share__}doc/root")
  else()
    set(CMAKE_INSTALL_DOCDIR ".")
  endif()
endif()

if(NOT CMAKE_INSTALL_TESTDIR)
  set(CMAKE_INSTALL_TESTDIR "" CACHE PATH "root tests (DOCDIR/test)")
  set(CMAKE_INSTALL_TESTDIR "${CMAKE_INSTALL_DOCDIR}/test")
endif()

if(NOT CMAKE_INSTALL_TUTDIR)
  set(CMAKE_INSTALL_TUTDIR "" CACHE PATH "root tutorials (DOCDIR/tutorials)")
  set(CMAKE_INSTALL_TUTDIR "${CMAKE_INSTALL_DOCDIR}/tutorials")
endif()


#-----------------------------------------------------------------------------

mark_as_advanced(
  CMAKE_INSTALL_BINDIR
  CMAKE_INSTALL_LIBDIR
  CMAKE_INSTALL_INCDIR
  CMAKE_INSTALL_ETCDIR
  CMAKE_INSTALL_MANDIR
  CMAKE_INSTALL_DATAROOTDIR
  CMAKE_INSTALL_DATADIR
  CMAKE_INSTALL_MACRODIR
  CMAKE_INSTALL_CINTINCDIR
  CMAKE_INSTALL_ICONDIR
  CMAKE_INSTALL_FONTDIR
  CMAKE_INSTALL_SRCDIR
  CMAKE_INSTALL_DOCDIR
  CMAKE_INSTALL_TESTDIR
  CMAKE_INSTALL_TUTDIR
  CMAKE_INSTALL_ACLOCALDIR
  CMAKE_INSTALL_ELISPDIR
  CMAKE_INSTALL_CMAKEDIR
  )

# Result directories
#
foreach(dir BINDIR 
            LIBDIR 
            INCDIR
            ETCDIR 
            MANDIR 
            DATAROOTDIR 
            DATADIR 
            MACRODIR 
            CINTINCDIR 
            ICONDIR 
            FONTDIR 
            SRCDIR 
            DOCDIR 
            TESTDIR 
            TUTDIR 
            ACLOCALDIR
            ELISPDIR 
            CMAKEDIR )
  if(NOT IS_ABSOLUTE ${CMAKE_INSTALL_${dir}})
    set(CMAKE_INSTALL_FULL_${dir} "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_${dir}}")
  else()
    set(CMAKE_INSTALL_FULL_${dir} "${CMAKE_INSTALL_${dir}}")
  endif()
endforeach()
