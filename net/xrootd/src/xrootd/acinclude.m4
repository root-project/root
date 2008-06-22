dnl @synopsis ACX_LOCATEFILE(filename,path-list,[action if found],[action if not found])
dnl
dnl    Locates a file in a given search path
dnl
dnl    the directory containing the target file is available as acx_founddir
dnl    the path-list is available as acx_searchpath
dnl
dnl    Author: Derek Feichtinger <derek.feichtinger@cern.ch>
dnl    
dnl    Version info: $Id: acinclude.m4,v 1.6 2008/05/19 11:36:17 dfeich Exp $
dnl    Checked in by $Author: dfeich $
dnl ========================================================================
AC_DEFUN([ACX_LOCATEFILE],[
  acx_searchpath="$2"
  acx_founddir=""
  AC_MSG_CHECKING([for $1 in $2])
  for dir in $2 ; do
     if test -f "$[]dir/$1" ; then
        acx_founddir="$[]dir/"
        break
     fi
  done
  if test x"$[]acx_founddir" = x ; then
    AC_MSG_RESULT([no])
    ifelse([$4], ,: ,[$4])
  else
    AC_MSG_RESULT([found])
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
	     acx_base_incdir="$withval/include"
	     acx_base_libdir="$withval/lib"
	     acx_base_bindir="$withval/bin"
             BASE_INCDIR="-I$withval/include"
             BASE_LIBDIR="-L$withval/lib"
            ]
           )
AC_SUBST(BASE_INCDIR)
AC_SUBST(BASE_LIBDIR)
])

dnl @synopsis ACX_COLLECT_OPTION(optionname,[optvalue])
dnl collects option names for an informative printout with
dnl the ACX_PRINTOPTIONS macro. Relies on the existence of
dnl an associated activate_FEATURE variable
AC_DEFUN([ACX_COLLECT_OPTION],[
   acx_optionvar="$[]acx_optionvar $1"
   ifelse($2, , :,activate_$1=$2)
#opt activate_$1=default_yes|default_no
])

dnl @synopsis ACX_LIBOPTION(optionname,enable-help-text,[yes|no])
dnl
dnl a default value for activate_FEATURE set before the evaluation of this
dnl macro will be honored.
dnl all options get collected in acx_optionvar for later printout with
dnl the ACX_PRINTOPTIONS macro
dnl specifying one of the --with-feature-*dir options will set the activate
dnl state of the feature to yes, except if it has been deliberately turned off
dnl with the --enable-feature=no or --disable-feature options
AC_DEFUN([ACX_LIBOPTION],[
   if test x"$[]activate_$1" = x; then 
      ifelse($3, yes,activate_$1=default_yes,activate_$1=default_no)
   fi
   AC_ARG_ENABLE($1,[[  --enable-$1        $2 (default=$3)]],[[activate_$1=$enableval]],[[:]])
   AC_ARG_WITH($1-libdir,[[  --with-$1-libdir=PATH    path containing $1 library]],
                [acx_$1_libdir=$[]withval
                if test $[]activate_$1 = default_no; then activate_$1=yes;fi
                if test $[]activate_$1 = default_yes; then activate_$1=yes;fi
   ])
   AC_ARG_WITH($1-incdir,[[  --with-$1-incdir=PATH    path containing $1 headers]],
                [acx_$1_incdir=$[]withval
                if test $[]activate_$1 = default_no; then activate_$1=yes;fi
                if test $[]activate_$1 = default_yes; then activate_$1=yes;fi
   ])

   if test x"$[]acx_$1_libdir" != x; then
      translit($1,`a-z',`A-Z')_LIBDIR="-L$[]acx_$1_libdir"
   fi
   if test x"$[]acx_$1_incdir" != x; then
      translit($1,`a-z',`A-Z')_INCDIR="-I$[]acx_$1_incdir"
   fi

   AC_SUBST(translit($1,`a-z',`A-Z')_LIBDIR)
   AC_SUBST(translit($1,`a-z',`A-Z')_INCDIR)

   ACX_COLLECT_OPTION($1)
#opt acx_$1_libdir=PATH
#opt acx_$1_incdir=PATH
])

dnl @synopsis ACX_PRINTOPTIONS
dnl
dnl prints a summary about option selections and default files
dnl that have been read.
dnl Relies on the oprions having been recorded by ACX_COLLECT_OPTION
dnl or ACX_LIBOPTION and default files having been read with ACX_LOAD_DEFAULTS
AC_DEFUN([ACX_PRINTOPTIONS],[
  echo
  if test x"$[]acx_defaultfiles" != x; then
     echo "Defaults read from: $[]acx_defaultfiles"
  fi

  echo "SELECTED OPTIONS"
  echo "----------------"
  for opt in $[]acx_optionvar; do
     optvar="activate_$[]opt"
     optval="\$$[]optvar"
     optval=`eval echo "$[]optval"`

     optinc=acx_"$[]opt"_incdir
     optincval="\$$[]optinc"
     optincval=`eval echo "$[]optincval"`

     optlib=acx_"$[]opt"_libdir
     optlibval="\$$[]optlib"
     optlibval=`eval echo "$[]optlibval"`

     defaultval=`expr x"$[]optval" : 'xdefault_\(.*\)'`
     if test x"$[]defaultval" != x; then
        optval=$[]defaultval
        bydefault="(by default)"
     else
        bydefault="            "
     fi

     if test x"$[]optval" != xyes; then
        optincval="";optlibval=""
     fi
     echo "$[]opt:     $[]optval   $[]bydefault  $[]optincval  $[]optlibval"
  done
])

dnl @synopsis ACX_LOAD_DEFAULTS(filename)
dnl
dnl Loads (sources) a file containing default settings.
dnl The location of the file has to be given relative
dnl to the srcdir
AC_DEFUN([ACX_LOAD_DEFAULTS],[
  acx_tmp="$[]srcdir/$1"
  AC_MSG_CHECKING([for defaults file $[]acx_tmp])
  if test -r "$[]acx_tmp"; then
     AC_MSG_RESULT([found - sourcing...])
     source $[]acx_tmp
     acx_defaultfiles="$acx_defaultfiles $[]acx_tmp"
  else
     AC_MSG_RESULT([NOT FOUND])
  fi
])


dnl AX_HAVE_EPOLL([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl AX_HAVE_EPOLL_PWAIT([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl (c) 2008 Peter Simons <simons@cryp.to>
dnl http://autoconf-archive.cryp.to/ax_have_epoll.html
dnl
dnl modified the acceptable Unix version for AX_HAVE_EPOLL
dnl to be 2,6,12 (original macro had 2.5.45)
AC_DEFUN([AX_HAVE_EPOLL], [dnl
  ax_have_epoll_cppflags="${CPPFLAGS}"
  AC_CHECK_HEADER([linux/version.h], [CPPFLAGS="${CPPFLAGS} -DHAVE_LINUX_VERSION_H"])
  AC_MSG_CHECKING([for Linux epoll(7) interface])
  AC_CACHE_VAL([ax_cv_have_epoll], [dnl
    AC_LINK_IFELSE([dnl
      AC_LANG_PROGRAM([dnl
#include <sys/epoll.h>
#ifdef HAVE_LINUX_VERSION_H
#  include <linux/version.h>
#  if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,12)
#    error linux kernel version is too old to have epoll
#  endif
#endif
], [dnl
int fd, rc;
struct epoll_event ev;
fd = epoll_create(128);
rc = epoll_wait(fd, &ev, 1, 0);])],
      [ax_cv_have_epoll=yes],
      [ax_cv_have_epoll=no])])
  CPPFLAGS="${ax_have_epoll_cppflags}"
  AS_IF([test "${ax_cv_have_epoll}" = "yes"],
    [AC_MSG_RESULT([yes])
$1],[AC_MSG_RESULT([no])
$2])
])dnl

AC_DEFUN([AX_HAVE_EPOLL_PWAIT], [dnl
  ax_have_epoll_cppflags="${CPPFLAGS}"
  AC_CHECK_HEADER([linux/version.h],
    [CPPFLAGS="${CPPFLAGS} -DHAVE_LINUX_VERSION_H"])
  AC_MSG_CHECKING([for Linux epoll(7) interface with signals extension])
  AC_CACHE_VAL([ax_cv_have_epoll_pwait], [dnl
    AC_LINK_IFELSE([dnl
      AC_LANG_PROGRAM([dnl
#ifdef HAVE_LINUX_VERSION_H
#  include <linux/version.h>
#  if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,19)
#    error linux kernel version is too old to have epoll_pwait
#  endif
#endif
#include <sys/epoll.h>
#include <signal.h>
], [dnl
int fd, rc;
struct epoll_event ev;
fd = epoll_create(128);
rc = epoll_wait(fd, &ev, 1, 0);
rc = epoll_pwait(fd, &ev, 1, 0, (sigset_t const *)(0));])],
      [ax_cv_have_epoll_pwait=yes],
      [ax_cv_have_epoll_pwait=no])])
  CPPFLAGS="${ax_have_epoll_cppflags}"
  AS_IF([test "${ax_cv_have_epoll_pwait}" = "yes"],
    [AC_MSG_RESULT([yes])
$1],[AC_MSG_RESULT([no])
$2])
])dnl


dnl AC_SYS_DEV_POLL([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
dnl Dave Benson <daveb@ffem.org> 2008-04-12
dnl from http://autoconf-archive.cryp.to/ac_sys_dev_poll.html
dnl
AC_DEFUN([AC_SYS_DEV_POLL], [AC_CACHE_CHECK(for /dev/poll support, ac_cv_dev_poll,
    AC_TRY_COMPILE([#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/poll.h>
#include <sys/devpoll.h>],
[
  struct dvpoll p;
  p.dp_timeout = 0;
  p.dp_nfds = 0;
  p.dp_fds = (struct pollfd *) 0;
  return 0;
],
    ac_cv_dev_poll=yes
    [$1],
    ac_cv_dev_poll=no
    [$2]
    )
  )
])
