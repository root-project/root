dnl
dnl $Id$
dnl $Author$
dnl $Date$
dnl

AC_DEFUN(ROOT_PATH,
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

    changequote(<<, >>)dnl
    ROOTCINT=`dirname $ROOTCONF`
    ROOTCINT="${ROOTCINT}/rootcint"
    ROOTLIBDIR=`$ROOTCONF --libdir`
    ROOTINCDIR=`$ROOTCONF --incdir`
    ROOTCFLAGS=`$ROOTCONF --noauxcflags --cflags` 
    ROOTLIBS=`$ROOTCONF --noauxlibs --noldflags --libs`
    ROOTGLIBS=`$ROOTCONF --noauxlibs --noldflags --glibs`
    ROOTAUXCFLAGS=`$ROOTCONF --auxcflags`
    ROOTAUXLIBS=`$ROOTCONF --auxlibs`
    changequote([, ])dnl

    AC_SUBST(ROOTCINT)
    AC_SUBST(ROOTLIBDIR)
    AC_SUBST(ROOTINCDIR)
    AC_SUBST(ROOTCFLAGS)
    AC_SUBST(ROOTLIBS)
    AC_SUBST(ROOTGLIBS) 
    AC_SUBST(ROOTAUXLIBS)
    AC_SUBST(ROOTAUXCFLAGS)
    AC_SUBST(ROOTRPATH)
])

