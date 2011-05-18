dnl @synopsis PFK_HAVE_GSL [--with-minuit-include=DIR] [--with-minuit-lib=DIR]
dnl
dnl Search for minuit libraries
dnl
dnl Searches common directories for minuit include files and library python. 
dnl If one of the options is given, the search is
dnl started in the given directory. If both options are given, no checks
dnl are performed and the cache is not consulted nor altered.
dnl
dnl The following shell variable is set to either "yes" or "no":
dnl
dnl   have_minuit
dnl
dnl Additionally, the following variables are exported:
dnl
dnl   GSL_CXXFLAGS
dnl   GSL_LDFLAGS
dnl
dnl which contain an "-I" flag pointing to the Qt include directory, link flags
dnl necessary to link with minuit.
dnl
dnl Example lines for Makefile.in:
dnl
dnl   INCLUDES = @GSL_CXXFLAGS@
dnl
dnl After the variables have been set, a trial compile and link is
dnl performed to check the correct functioning of the meta object compiler.
dnl If this test fails, a warning appears in the output of configure, but
dnl the variables remain defined.
dnl
dnl No common variables such as $LIBS or $CFLAGS are polluted.
dnl
dnl Options:
dnl
dnl --with-gsl-include=DIR: DIR should point to where the gsl
dnl include directory can be found, as in -IDIR.
dnl
dnl --with-gsl-lib=DIR: DIR should point to the where tohe libraries 
dnl can be found as in -LDIR
dnl
dnl If at least one of the options "=no" or, equivalently,
dnl --without-gsl-include and/or --without-gsl-lib is given, 
dnl "have_gsl" is set to "no" and the other variables are set to the 
dnl empty string.
dnl
dnl Calls PFK_HAVE_GSL_LDFLAGSECT as a subroutine.
dnl
dnl @version $Id: ac_have_gsl.m4,v 1.3 2004/11/24 16:38:49 moneta Exp $
dnl @author <Paul_Kunz@slac.stanford.edu> base on bnv_have_qt.mr by
dnl Bastiaan N. Veelo <Bastiaan.N.Veelo@immtek.ntnu.no>
dnl
AC_DEFUN([AC_HAVE_GSL],
[
  AC_REQUIRE([AC_PROG_CXX])
  have_gsl="yes"
	
  AC_ARG_WITH([gsl],
    [  --with-gsl=DIR root directory of gsl installation 
                                  ])
  if test x"$with_gsl" != x; then  
    with_gsl_include=$with_gsl/include
    with_gsl_lib=$with_gsl/lib
  fi

  AC_ARG_WITH([gsl-include],
    [  --with-gsl-include=DIR gsl include directory is in DIR 
                                  ])
  AC_ARG_WITH([gsl-lib],
    [  --with-gsl-lib=DIR      gsl libgsl.a file is in DIR 
                                  ])
  if test x"$with_gsl_include" = x"no" || 
     test x"$with_gsl_lib" = x"no"; then
    dnl user disabled gsl. Leave cache alone.
    have_gsl="no"
  fi
    # "yes" is a bogus option
  if test "x$with_gsl_include" = xyes; then
    with_gsl_include=
  fi
  if test "x$with_gsl_lib" = xyes; then
    with_gsl_lib=
  fi

dnl  set defaults
dnl PFK_PATH_GSL_DIRECT

  if test "x$with_gsl_include" != x; then
    pfk_gsl_include="$with_gsl_include"
  fi
  if test "x$with_gsl_lib" != x; then
    pfk_gsl_lib="$with_gsl_lib"
  fi

  if test x"$have_gsl" = xyes; then
    AC_CHECK_FILE($pfk_gsl_include/gsl/gsl_math.h,, no_good=yes )
    AC_CHECK_FILE($pfk_gsl_lib/libgsl.a,, no_good=yes )
  fi
  if test x$no_good = xyes; then
    have_gsl=no
  fi
  if test x"$have_gsl" = xyes; then
    GSL_CXXFLAGS="-I$pfk_gsl_include"
    GSL_LDFLAGS="-L$pfk_gsl_lib -lgsl -lgslcblas"
    AC_MSG_RESULT([gsl found
    GSL_CXXFLAGS=$GSL_CXXFLAGS
    GSL_LDFLAGS=$GSL_LDFLAGS ] )
dnl    AC_DEFINE(HAVE_GSL, 1,
dnl      Define to 1 if C++ version of gsl is available )
  else
    GSL_CXXFLAGS=
    GSL_LDFLAGS= 
    AC_MSG_WARN([ gsl not found - use local installation from ../gsl 
    GSL_CXXFLAGS=$GSL_CXXFLAGS
    GSL_LDFLAGS=$GSL_LDFLAGS ] )
  fi
  AC_SUBST(GSL_CXXFLAGS) AC_SUBST(GSL_LDFLAGS)
])

dnl Internal subroutine of PFK_HAVE_GSL
dnl AC_DEFUN(PFK_PATH_GSL_DIRECT,
dnl [
dnl pfk_gsl_include="/usr/local/include"
dnl pfk_gsl_lib="/usr/local/lib"
dnl ])
