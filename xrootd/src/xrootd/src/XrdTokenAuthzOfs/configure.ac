#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT(XrdTokenAuthzOfs, 1.0.0,[xrootd-l@slac.stanford.edu])
AC_CONFIG_SRCDIR([XrdTokenAuthzOfs.cc])

AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE(1.6 foreign)

case "$target_os" in
   tiger*)
     AC_DEFINE(__macos__)
     ;;
   darwin*)
     AC_DEFINE(__macos__)
     ;;
   apple*)
     AC_DEFINE(__macos__)
     ;;
   linux*)
     # TODO: check why two different macros are used for this
     AC_DEFINE(__linux__)
     AC_DEFINE(__linux)
     ;;
   solaris*)
     AC_DEFINE(__sun)
     CC_PREFERENCE="cc gcc"
     CXX_PREFERENCE="CC g++"
     DFLT_OPT="-O2"
     ;;
   *)
     AC_MSG_WARN([untested operating system])
   ;;
esac


ACX_WITH_BASEDIR

AC_ARG_WITH(xrootd-src,
           [  --with-xrootd-src=PATH   directory with xrootd compiled sources],
            [XROOTD_SRCDIR="$withval"]
           )
if test x"$XROOTD_SRCDIR" = x ; then
   XROOTD_SRCDIR="../.."
fi
if test ! -f "$XROOTD_SRCDIR/src/XrdOfs/XrdOfs.cc" ; then
   AC_MSG_ERROR([Could not locate src/XrdOfs/XrdOfs.cc in $XROOTD_SRCDIR - use --with-xrootd-src option])
fi
AC_SUBST(XROOTD_SRCDIR)

# We need to find out whether we are built on the new or on the old
# xrootd tarball. We just check for libtool
if test -f "$XROOTD_SRCDIR/ltmain.sh" ; then
  AC_MSG_NOTICE([We are building on top of an autotool build of xrootd])
  xrootd_autobuild=yes
else
  AC_MSG_NOTICE([We are building on top of an non autotool build of xrootd])
  XROOTD_LIBDIR="$XROOTD_SRCDIR/lib/arch"
  AC_SUBST(XROOTD_LIBDIR)
fi

AM_CONDITIONAL(XROOTD_AUTOBUILD, test x"$xrootd_autobuild" = xyes)


AC_ARG_WITH(tkauthz-inc,
            [  --with-tkauthz-inc=PATH    directory containing TTokenAuthz.h],
            [TKAUTHZ_INC="-I${withval}"]
           )
AC_SUBST(TKAUTHZ_INC)
AC_ARG_WITH(tkauthz-lib,
            [  --with-tkauthz-lib=PATH    directory containing libTokenAuthz],
            [TKAUTHZ_LIB="-L${withval}"]
           )
AC_SUBST(TKAUTHZ_LIB)

CPPFLAGS_BUP="$CPPFLAGS"
LDFLAGS_BUP="$LDFLAGS"


# Do all testing using C++
AC_LANG([C++])

# Checks for programs.
AC_PROG_CXX

dnl Disable libtool 1.5 support for languages we don't use
define([AC_LIBTOOL_LANG_CXX_CONFIG], [:])dnl
define([AC_LIBTOOL_LANG_F77_CONFIG], [:])dnl
define([AC_LIBTOOL_LANG_GCJ_CONFIG], [:])dnl
AC_PROG_LIBTOOL

# Checks for libraries.
LDFLAGS="$TKAUTHZ_LIB $LDFLAGS"
AC_CHECK_LIB([TokenAuthz],[main],
             [:],[ACX_MSG_ERROR([failed library test for libTokenAuthz]) ]
            )
LDFLAGS="$LDFLAGS_BUP"


# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([sys/time.h])

CPPFLAGS="$TKAUTHZ_INC $CPPFLAGS"
AC_CHECK_HEADER([TTokenAuthz.h],
                [],[ACX_MSG_ERROR([failed include test for TTokenAuthz.h]) ]
               )
CPPFLAGS="$CPPFLAGS_BUP"


# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_MODE_T
AC_TYPE_SIZE_T
AC_HEADER_TIME

# Checks for library functions.
AC_FUNC_STAT
AC_CHECK_FUNCS([gettimeofday mkdir strdup])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
