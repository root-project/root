#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# 
# Author: Stefan Roiser <stefan.roiser@cern.ch>
#

# general 
AC_INIT(reflex, 5.08.00.2, http://cern.ch/seal-reflex, Reflex)
AC_PREREQ(2.57) # Minimum version of autoconf which is required
AC_COPYRIGHT(This package is part of the CERN/LCG project)
AC_CONFIG_SRCDIR([src/Class.cxx]) # A unique file in the src dir
AC_CONFIG_AUX_DIR(config)
AM_CONFIG_HEADER([inc/Reflex/config.h])
AM_INIT_AUTOMAKE(dist-bzip2 dist-zip)

# libtool
AC_LIBLTDL_CONVENIENCE # install libtool and libltdl locally
AC_SUBST([LDTINCL])
AC_SUBST([LIBLTDL]) 
AC_LIBTOOL_DLOPEN # check for dlopen support
AC_PROG_LIBTOOL # enable libtool

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
# gccxml testing
AM_CONDITIONAL([GCCXMLROOT],[test "$GCCXMLROOT" != 0])
AC_ARG_VAR([GCCXMLROOT],[The root directory of the gccxml tool])
AC_SUBST([GCCXMLROOT])
# cppunit testing
AM_CONDITIONAL([CPPUNITROOT],[test "$CPPUNITROOT" != 0])
AC_ARG_VAR([CPPUNITROOT],[The root directory of the cppunit test suite])
AC_SUBST([CPPUNITROOT])

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_C_VOLATILE
AC_C_BIGENDIAN # defines WORDS_BIGENDIAN if big endian architecture
AC_C_CHAR_UNSIGNED # defines __CHAR_UNSIGNED__ if char is unsigned
AC_C_LONG_DOUBLE # defines HAVE_LONG_DOUBLE if more precision than double

# Checks for library functions.
AC_HEADER_STDC # defines STDC_HEADERS if ANSI headers available

# Checks for python
AM_PATH_PYTHON

# configuration of tests
AC_ARG_ENABLE([tests],AC_HELP_STRING([--enable-tests],[enable compilation of the test suite]))

DIST_REFLEXTESTDIRS="test"
AC_SUBST([DIST_REFLEXTESTDIRS])

LIBLTDLDIR=
AC_SUBST([LIBLTDLDIR])

if [[ "$enable_tests" == "yes" ]]; then
  AC_CONFIG_SUBDIRS([ libltdl ])
  LIBLTDLDIR="libltdl"
  if (test "$GCCXMLROOT") && (test "$CPPUNITROOT"); then
    REFLEXTESTDIRS=$DIST_REFLEXTESTDIRS
  else
    REFLEXTESTDIRS=
  fi
fi
AC_SUBST([REFLEXTESTDIRS])

# config files and output 
AC_CONFIG_FILES([ Makefile                     \
                  doc/Makefile                 \
                  inc/Makefile                 \
                  inc/Reflex/Makefile          \
                  inc/Reflex/Builder/Makefile  \
                  inc/Reflex/internal/Makefile \
                  python/Makefile              \
                  python/genreflex/Makefile    \
                  src/Makefile                 \
                  test/Makefile                \
                  test/testDict1/Makefile      \
                  test/testDict2/Makefile
                ])

case ${host} in 
*-*-windows*) pf=win;;
*) pf=lin;;
esac
AM_CONDITIONAL(PLATFORM_IS_WINDOWS,test x$pf = xwin)

AC_OUTPUT

# print warning if tests are enabled but gccxml not found
if ([[ "$enable_tests" == "yes" ]]) && ( (! test "$GCCXMLROOT") || (! test "$CPPUNITROOT") ) ; then
AC_MSG_WARN([ 
*********************************************************************************
  !!!!    There is a problem with the current configuration environment    !!!!

To run the tests of this package, the following variable(s) need to be defined :

GCCXMLROOT  : The root directory of the gccxml installation
CPPUNITROOT : The root directory of the cppunit test suite

The variables can be defined as environment variables (eg. with tcsh "setenv") or
passed to the configure command (e.g. ./configure GCCXMLROOT=/usr/local/gccxml ).
The variables should point to the directory which contain /bin,/lib,etc. subdirs.
*********************************************************************************
]) 
fi
