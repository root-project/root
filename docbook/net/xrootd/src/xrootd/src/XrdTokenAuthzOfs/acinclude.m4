dnl @synopsis ACX_LOCATEFILE(filename,path-list,[action if found],[action if not found])
dnl
dnl    Locates a file in a given search path
dnl
dnl    the directory containing the target file is available as acx_founddir
dnl    the path-list is available as acx_searchpath
dnl
dnl    Author: Derek Feichtinger <derek.feichtinger@cern.ch>
dnl    
dnl    Version info: $Id$
dnl    Checked in by $Author$
dnl ========================================================================
AC_DEFUN([ACX_LOCATEFILE],[
acx_searchpath="$2"
acx_founddir=""
for dir in $2 ; do
   if test -f "$[]dir/$1" ; then
      acx_founddir="$[]dir/"
      break
   fi
done
if test x"$[]acx_founddir" = x ; then
  ifelse([$4], ,: ,[$4])
else
  ifelse([$3], ,: ,[$3])
fi

])

dnl @synopsis ACX_MSG_ERROR(error-message)
dnl
dnl    like AC_MSG_ERROR, but also prints out some important
dnl    environment settings
AC_DEFUN([ACX_MSG_ERROR],[
   AC_MSG_ERROR([$1
  (CPPFLAGS="$[]CPPFLAGS)   (LDFLAGS="$[]LDFLAGS")
  (CFLAGS="$[]CFLAGS")   (CXXFLAGS="$[]CXXFLAGS")])
])

dnl @synopsis ACX_WITH_BASEDIR
dnl ############################################################
dnl # give the user an easy way to specify a base installation
dnl # directory dir, where headers and libraries are found in
dnl # $dir/include and $dir/lib
AC_DEFUN([ACX_WITH_BASEDIR],[
AC_ARG_WITH(base-directory,
            [  --with-base-directory=PATH   add PATH/include and PATH/lib to search paths],
            [
             BASE_INCDIR="$withval/include"
             BASE_LIBDIR="$withval/lib"
             CPPFLAGS="-I$BASE_INCDIR $CPPFLAGS"
             LDFLAGS="-L$BASE_LIBDIR $LDFLAGS"
            ]
           )
AC_SUBST(BASE_INCDIR)
AC_SUBST(BASE_LIBDIR)
])