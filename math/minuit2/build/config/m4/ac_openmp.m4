# openmp.m4 serial 4
dnl Copyright (C) 2006-2007 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.

dnl This file can be removed once we assume autoconf >= 2.62.

# _AC_LANG_OPENMP
# ---------------
# Expands to some language dependent source code for testing the presence of
# OpenMP.
AC_DEFUN([_AC_LANG_OPENMP],
[_AC_LANG_DISPATCH([$0], _AC_LANG, $@)])

# _AC_LANG_OPENMP(C)
# ------------------
m4_define([_AC_LANG_OPENMP(C)],
[
#ifndef _OPENMP
 choke me
#endif
#include <omp.h>
int main () { return omp_get_num_threads (); }
])

# _AC_LANG_OPENMP(C++)
# --------------------
m4_copy([_AC_LANG_OPENMP(C)], [_AC_LANG_OPENMP(C++)])

# _AC_LANG_OPENMP(Fortran 77)
# ---------------------------
m4_define([_AC_LANG_OPENMP(Fortran 77)],
[AC_LANG_FUNC_LINK_TRY([omp_get_num_threads])])

# _AC_LANG_OPENMP(Fortran)
# ---------------------------
m4_copy([_AC_LANG_OPENMP(Fortran 77)], [_AC_LANG_OPENMP(Fortran)])

# AC_OPENMP
# ---------
# Check which options need to be passed to the C compiler to support OpenMP.
# Set the OPENMP_CFLAGS / OPENMP_CXXFLAGS / OPENMP_FFLAGS variable to these
# options.
# The options are necessary at compile time (so the #pragmas are understood)
# and at link time (so the appropriate library is linked with).
# This macro takes care to not produce redundant options if $CC $CFLAGS already
# supports OpenMP. It also is careful to not pass options to compilers that
# misinterpret them; for example, most compilers accept "-openmp" and create
# an output file called 'penmp' rather than activating OpenMP support.
AC_DEFUN([AC_OPENMP],
[
  OPENMP_[]_AC_LANG_PREFIX[]FLAGS=
  AC_ARG_ENABLE([openmp],
    [AS_HELP_STRING([--disable-openmp], [do not use OpenMP])])
  if test "$enable_openmp" != no; then
    AC_CACHE_CHECK([for $CC option to support OpenMP],
      [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp],
      [AC_LINK_IFELSE([_AC_LANG_OPENMP],
	 [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp='none needed'],
	 [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp='unsupported'
	  dnl Try these flags:
	  dnl   GCC >= 4.2           -fopenmp
	  dnl   SunPRO C             -xopenmp
	  dnl   Intel C              -openmp
	  dnl   SGI C, PGI C         -mp
	  dnl   Tru64 Compaq C       -omp
	  dnl   IBM C (AIX, Linux)   -qsmp=omp
	  dnl If in this loop a compiler is passed an option that it doesn't
	  dnl understand or that it misinterprets, the AC_LINK_IFELSE test
	  dnl will fail (since we know that it failed without the option),
	  dnl therefore the loop will continue searching for an option, and
	  dnl no output file called 'penmp' or 'mp' is created.
	  for ac_option in -fopenmp -xopenmp -openmp -mp -omp -qsmp=omp; do
	    ac_save_[]_AC_LANG_PREFIX[]FLAGS=$[]_AC_LANG_PREFIX[]FLAGS
	    _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $ac_option"
	    AC_LINK_IFELSE([_AC_LANG_OPENMP],
	      [ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp=$ac_option])
	    _AC_LANG_PREFIX[]FLAGS=$ac_save_[]_AC_LANG_PREFIX[]FLAGS
	    if test "$ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp" != unsupported; then
	      break
	    fi
	  done])])
    case $ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp in #(
      "none needed" | unsupported)
        ;; #(
      *)
        OPENMP_[]_AC_LANG_PREFIX[]FLAGS=$ac_cv_prog_[]_AC_LANG_ABBREV[]_openmp ;;
    esac
  fi
  AC_SUBST([OPENMP_]_AC_LANG_PREFIX[FLAGS])
])
