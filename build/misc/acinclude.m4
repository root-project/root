dnl
dnl $Id: acinclude.m4,v 1.1.1.1 2000/11/28 17:18:40 cholm Exp $
dnl $Author: cholm $
dnl $Date: 2000/11/28 17:18:40 $
dnl
AC_DEFUN(AC_ROOT,
[
  AC_ARG_WITH(rootsys,
    [  --with-rootsys          top of the ROOT installation directory],
      user_rootsys=$withval,
      user_rootsys="none")
  if test ! x"$user_rootsys" = xnone; then
    rootbin="$user_rootsys/bin"
  elif test ! x"$ROOTSYS" = x ; then 
    rootbin="$ROOTSYS/bin"
  else 
    rootbin=$PATH
  fi
  AC_PATH_PROG(ROOTCONF, root-config , no, $rootbin)
  if test x"$ROOTCONF" = "xno" ; then
    AC_MSG_ERROR([ROOT config script not found!])
  fi
])

dnl
dnl $Log: acinclude.m4,v $
dnl
