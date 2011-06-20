#######################################################################
#                      configure.ac
# Process this file with autoconf to produce a configure script.
#######################################################################
##
## (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University
##       All Rights Reserved. See XrdInfo.cc for complete License Terms
##   Produced by Andrew Hanushevsky for Stanford University under contract
##              DE-AC03-76-SFO0515 with the Department of Energy
##
## configure.ac autotools configuration file for xrootd
## Author: Derek Feichtinger <derek.feichtinger@cern.ch>
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

# Note: All macros with the prefix ACX_ are custom macros. Their usage
# information and the source code can be found in acinclude.m4


AC_PREREQ(2.57)
AC_INIT(xrootd,3.0.0,xrootd-l@slac.stanford.edu)
AC_CONFIG_SRCDIR([src/XrdVersion.hh.in])

# use C++ for the compilation of tests
dnl currently commented out, since older versions of libtool (<1.5) contain
dnl a test similar to AM_PROG_CC_C_O which triggers locking by libtool itself
dnl if AC_LANG(C++) is in effect.
dnl Since the locking is implemented via a hardlink to a file, this does not
dnl work on AFS
dnl AC_LANG(C++)
AC_LANG(C++)

###############################################################
# Set some defaults which will be augmented by the OS detection
# routine below

# extra directories for standard search path for header files
extra_inc_path_default=" "

# extra directories for standard search path for libraries
extra_lib_path_default=" "

# preference for compilers
CC_PREFERENCE="gcc cc"
CXX_PREFERENCE="g++ c++"

# Note:
# adding wildly to CFLAGS, CXXFLAGS, LDFLAGS, etc is not good
# style. The user should have the possibility to use
# the more specific --with-FEATURE-DIR options to specify locations.
# Any explicitely specified LDFLAGS (e.g. LDFLAGS="-Lpath") will be put
# in front of other options and therefore may override them!

###############################################################

# detect system type and define some specific settings
AC_CANONICAL_TARGET
case "$target_os" in
   tiger*)
     AC_DEFINE(__macos__)
     ;;
   darwin*)
     AC_DEFINE(__macos__)
     need_macos_dt=no
     if test x"$MACOSX_DEPLOYMENT_TARGET" = x; then
        macosversion=`sw_vers -productVersion`
        AC_MSG_NOTICE([Detected MacOS Version $macosversion])
        vermajor=`expr x"$macosversion" : 'x\([[0-9]][[0-9]]*\).[[0-9]][[0-9]]*'`
        verminor=`expr x"$macosversion" : 'x[[0-9]][[0-9]]*.\([[0-9]][[0-9]]*\)'`
        AC_MSG_WARN([MACOSX_DEPLOYMENT_TARGET environment variable not set. Setting it to ${vermajor}.$verminor])
        AC_SUBST([MACOSX_DEPLOYMENT_TARGET],${vermajor}.$verminor)
        need_macos_dt=yes
     else
        # if the user has set the variable himself, we trust him
        vermajor=`expr x"$MACOSX_DEPLOYMENT_TARGET" : 'x\([[0-9]][[0-9]]*\).[[0-9]][[0-9]]*'`
        verminor=`expr x"$MACOSX_DEPLOYMENT_TARGET" : 'x[[0-9]][[0-9]]*.\([[0-9]][[0-9]]*\)'`
        if test 0"$vermajor" -lt 10 ; then need_macos_dt=ok;
        else if test 0"$vermajor" -eq 10 -a 0"$verminor" -lt 3; then need_macos_dt=failed;
              fi
        fi
        AC_SUBST([MACOSX_DEPLOYMENT_TARGET],${MACOSX_DEPLOYMENT_TARGET})

     fi
     if test x"$need_macos_dt" = xfailed; then
        AC_MSG_ERROR([MACOSX_DEPLOYMENT_TARGET must be >= 10.3 (current value: $MACOSX_DEPLOYMENT_TARGET)])
     fi
     ;;
   apple*)
     AC_DEFINE(__macos__)
     ;;
   linux*)
     AC_DEFINE(__linux__)
     if test -r /etc/redhat-release; then
        # load the redhat linux default option settings
        ACX_LOAD_DEFAULTS(config/ac_default_linux_rh)
     fi
     ;;
   solaris*)
     TARGET_OS=solaris
     AC_DEFINE(__solaris__)
     CC_PREFERENCE="cc gcc"
     CXX_PREFERENCE="CC g++"
     DFLT_OPT="-O2"
     extra_inc_path_default="$extra_inc_path_default /usr/local/include"
     extra_lib_path_default="$extra_lib_path_default /usr/local/lib"
     ;;
   *)
     AC_MSG_WARN([untested operating system])
   ;;
esac

case "$target_cpu" in
   i386)
     AC_DEFINE(__i386__)
     ;;
   *)
     ;;
esac

AM_CONDITIONAL(NEED_MACOS_DEPLOYMENT_TARGET,[test x"$need_macos_dt" = xyes])

###########################################################
# We allow site specific default configurations to be read from files in the
# config area (important for people at SLAC where some depending
# libraries are located in non standard places, and people do not want
# every time to specify long configure options)
domain=""
case "$target_os" in
   linux*)
     domain=`dnsdomainname 2> /dev/null`
     ;;
   *)
     domain=`domainname 2> /dev/null`
     ;;
esac
if test "x$domain" = "x" ; then
   domain=`domainname 2> /dev/null`
fi

# look whether there is a site specific defaults file for this domain
### TODO: does this line still make sense?    domain=`expr x"$domain" : 'x.*\.\(.*\..*\)$'`
if test x"$domain" != x -a  x"$domain" != "x(none)"; then
   ACX_LOAD_DEFAULTS(config/ac_default_$domain)
fi

# Allow loading of a user specified defaults file
AC_ARG_WITH(defaults,
	[  --with-defaults=PATH        path to a default options file to be sourced],
        [if test x"$withval" = x -o ! -r "$withval"; then
            AC_MSG_ERROR([Cannot open specified defaults file ($withval)])
         fi
         ACX_LOAD_DEFAULTS($withval)
         ])

###########################################################
# only carry on the paths which really exist on this system
for name in $extra_inc_path_default ; do
   if test -d "$name"; then
      extra_inc_path="$extra_inc_path $name"
   fi
done
for name in $extra_lib_path_default ; do
   if test -d "$name"; then
      extra_lib_path="$extra_lib_path $name"
   fi
done


# this extension allows the definition of a base dir for finding dependencies
# in basedir/lib, basedir/etc, ...
# This is easier for the user than forcing him to specify multiple
# --with-FEATURE-DIR options or having him set CPPFLAGS/LDFLAGS
ACX_WITH_BASEDIR

extra_inc_path="$acx_base_incdir $extra_inc_path"
extra_lib_path="$acx_base_libdir $extra_lib_path"


############################################################
# set some variables according to what has been found so far

# for some tests we need the standard include path.
# We need to add any settings the user may have set via CPPFLAGS
for opt in $CPPFLAGS ; do
   case $opt in
      -I*)
         dir=`expr x"$opt" : 'x-I\(.*\)'`
         std_inc_path="$std_inc_path $dir"
         ;;
      *)
        ;;
   esac
done
std_inc_path="$std_inc_path $extra_inc_path /usr/include"

# add extra include path elements to CPPFLAGS
for dir in $extra_inc_path ; do
   if test x"$dir" != x ; then
      CPPFLAGS="$CPPFLAGS -I$dir"
   fi
done

# add extra library path elements to LDFLAGS
for dir in $extra_lib_path ; do
   if test x"$dir" != x ; then
      LDFLAGS="$LDFLAGS -L$dir"
   fi
done

AC_MSG_NOTICE([my CPPFLAGS... $CPPFLAGS])
AC_MSG_NOTICE([my LDFLAGS... $LDFLAGS])

############################################################
# BUILDING WITH DEBUG SYMBOLS

AC_ARG_ENABLE(dbg-build,
              [  --enable-dbg-build      build with debug symbols, no optimization],
  [if test x"$enableval" = "xyes"; then
     DFLT_OPT=""
     DBG_OPT="-g"
     AC_MSG_NOTICE([ENABLING DEBUG BUILD])
  fi],[enableval=default_no]
)
ACX_COLLECT_OPTION(dbg_build,$enableval)


#################################################################
# Setting optimization and debug compiler options
#    Note: needs to be set before calling AC_PROG_CXX, because
#    if CXXFLAGS is empty, that macro will fill it with the
#    autoconf default (which is -g for any non-gcc compiler)

if test x"$DBG_OPT""$DFLT_OPT" != x ; then
   CXXFLAGS="$DBG_OPT $DFLT_OPT $CXXFLAGS"
fi

# get compiler information
AC_PROG_CC([$CC_PREFERENCE])
AC_PROG_CXX([$CXX_PREFERENCE])
AM_CONDITIONAL(IS_GCC,[test x"$GCC" = xyes])

#   Note: necessary for older distributions (e.g. RH7.3): sets a number of C standards macros
if test x"$GCC" = xyes ; then
   AC_GNU_SOURCE
fi

# Check for GCC's new atomic ops (as of 4.1.2)
AC_CACHE_CHECK([whether gcc provide atomic ops], [ac_atomic_ops],
[AC_TRY_RUN([
int main()
{
    unsigned long long val = 111, tmp, *mem = &val;

    if (__sync_fetch_and_add(&val, 111) != 111 || val != 222) return 1;
    if (__sync_add_and_fetch(&val, 111) != 333)               return 1;
    if (__sync_sub_and_fetch(&val, 111) != 222)               return 1;
    if (__sync_fetch_and_sub(&val, 111) != 222 || val != 111) return 1;

    if (__sync_fetch_and_or (&val, 0)   != 111 || val != 111) return 1;
    if (__sync_fetch_and_and(&val, 0)   != 111 || val != 0  ) return 1;

    if (__sync_bool_compare_and_swap(mem, 0, 444) == 0 || val != 444)
        return 1;

    return 0;
}], ac_atomic_ops=yes, ac_atomic_ops=no, ac_atomic_ops=no)])

if test "x$ac_atomic_ops" = "xyes"; then
  AC_DEFINE(HAVE_ATOMICS)
fi

# Need to find a cleaner solution for the large file support settings
# The standard autoconf macro AC_SYS_LARGEFILE is not enough for xrootd which
# explicitely tests for a number of compilation flags in XrdPosixExtern.hh
# For now, we set the flags explicitely based on the architecture
case "$target_os" in
   solaris*)
     AC_SYS_LARGEFILE
     ;;
   *)
     AC_DEFINE(_LARGEFILE_SOURCE)
     AC_DEFINE(_LARGEFILE64_SOURCE)
     AC_DEFINE(_FILE_OFFSET_BITS,64)
     ;;
esac

AM_INIT_AUTOMAKE(1.6 subdir-objects)
dnl We add the option to build silently (but only if the symbol is available
dnl    which should be supported by automake >= 1.11)
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES])


# we save the environment's value for CPPFLAGS,etc 
# because we will modify them for some of the tests
CPPFLAGS_BUP="$CPPFLAGS"
LDFLAGS_BUP="$LDFLAGS"


# libsocket is needed on solaris
AC_CHECK_LIB([socket],[socket],
               [LIBSOCKET="-lsocket"],[LIBSOCKET=""],[]
              )
AC_SUBST(LIBSOCKET)


###################################################################
# OPTIONAL PACKAGES AND DIRECTIVES
###################################################################


####################################################################
# passwd based authentication support
if test x"$activate_pwd" = x; then
   activate_pwd=default_yes
fi
AC_ARG_ENABLE(pwd,
              [  --enable-pwd            Password-based authentication support [[default=yes]] ],
              [],[enable_pwd=$activate_pwd])
AM_CONDITIONAL(ENABLE_PWD, test x"$enable_pwd" = xyes -o x"$enable_pwd" = xdefault_yes)
if test x"$enable_pwd" = xyes -o x"$enable_pwd" = xdefault_yes; then
   if test -f "/etc/shadow"; then
     AC_DEFINE(R__SHADOWPW)
   fi
   AC_CHECK_LIB(crypt,[crypt],
                 [AC_SUBST(CRYPTLIB,[-lcrypt])],
		 [AC_DEFINE(DONT_HAVE_CRYPT)
		  AC_MSG_WARN([disabling support for crypt-like passwords])
		 ],[])
fi
ACX_COLLECT_OPTION(pwd,$enable_pwd)

####################################################################
# Kerberos IV support
ACX_LIBOPTION(krb4,[   Kerberos4 support],yes)
if test x"$acx_krb4_incdir" != x ; then
   AC_MSG_CHECKING([for kerberosIV/krb.h in $acx_krb4_incdir])
   if test ! -f "$acx_krb4_incdir"/kerberosIV/krb.h ; then
      AC_MSG_RESULT([no])
      if test $activate_krb4 = yes; then
         AC_MSG_ERROR([Could not locate kerberosIV/krb.h in $acx_krb4_incdir])
      else
         AC_MSG_WARN([disabling krb4 support])
         activate_krb4=disabled
      fi
   else
      AC_MSG_RESULT([found])
   fi
fi

####################################################################
# Kerberos 5 support
ACX_LIBOPTION(krb5,[   Kerberos5 support],yes)
if test x"$acx_krb5_incdir" != x ; then
   AC_MSG_CHECKING([for krb5.h in $acx_krb5_incdir])
   if test ! -f "$acx_krb5_incdir"/krb5.h ; then
      AC_MSG_RESULT([no])
      if test $activate_krb5 = yes; then
         AC_MSG_ERROR([Could not locate krb.h in $acx_krb5_incdir])
      else
         AC_MSG_WARN([disabling krb5 support])
         activate_krb5=disabled
      fi
   else      
      AC_MSG_RESULT([found])
   fi
fi

####################################################################
# GSI support
if test x"$activate_gsi" = x; then
   activate_gsi=default_no
fi
AC_ARG_ENABLE(gsi,
              [  --enable-gsi            GSI authentication support, requires OpenSSL [[default=no]] ],
              [],[enable_gsi=$activate_gsi]
              )
ACX_COLLECT_OPTION(gsi,$enable_gsi)
if test x"$activate_gsi" = xyes -o x"$activate_gsi" = xdefault_yes ; then
   AC_MSG_NOTICE([gsi module requires enabling of openssl module - enabling...])
   activate_openssl=default_yes
fi


####################################################################
# OpenSSL support
ACX_LIBOPTION(openssl,[Build OpenSSL implementation of crypto interfaces],yes)

if test x"$activate_openssl" = xyes -o x"$activate_openssl" = xdefault_yes ; then
  ssltests=ok

  if test x"$acx_openssl_incdir" != x; then
     searchpath=$acx_openssl_incdir
  else
     searchpath="$std_inc_path"
  fi

  ACX_LOCATEFILE([openssl/opensslv.h],[$searchpath],
                 [OPENSSL_INCDIR="-I$acx_founddir"
                  OPENSSL_VERSION_TEXT=`grep OPENSSL_VERSION_TEXT "$dir"/openssl/opensslv.h | grep OpenSSL`
                  AC_MSG_NOTICE([openssl header dir set to: $dir ]) ],
                 [if test  x"$activate_openssl" = xyes ; then
                    AC_MSG_ERROR([Could not locate openssl/opensslv.h])
                  else
                    activate_openssl=disabled
                  fi
                 ]
                )

  #AC_SUBST(OPENSSL_INCDIR)

  # test openssl version
  verstr=`expr  x"$OPENSSL_VERSION_TEXT" : '.*\([[0-9]][[0-9]]*\.[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*[[a-z]]*\)'`
  AC_MSG_NOTICE([OpenSSL version : $verstr])

  vermajor=`expr x"$verstr" : '.*\([[0-9]][[0-9]]*\).[[0-9]][[0-9]]*.[[0-9]][[0-9]]*'`
  verminor=`expr x"$verstr" : '.*[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\)\.[[0-9]][[0-9]]*'`
  verrelease=`expr x"$verstr" : '.*[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\)'`
  vernum=`expr 10000 \* 0"$vermajor" + 100 \* 0"$verminor" + 0"$verrelease"`
  if test 0"$vernum" -lt 906 ; then
    AC_MSG_WARN([OpenSSL >= 0.9.6 required (found: $OPENSSL_VERSION_TEXT)
                ssltests=failed])
  fi
  #AC_MSG_NOTICE([Openssl version: $vermajor  $verminor  $verrelease == $vernum])

  # note: openssl may require kerberos5 include files
  CPPFLAGS="$OPENSSL_INCDIR $KRB5_INCDIR $CPPFLAGS"
  AC_CHECK_HEADER([openssl/ssl.h],
                [],[ssltests=failed]
                )


  LDFLAGS="$OPENSSL_LIBDIR $LDFLAGS"
  AC_CHECK_LIB([ssl],[SSL_library_init],
               [:],[ssltests=failed],[-lcrypto $LIBSOCKET]
              )

  if test x"$ssltests" != xok ; then
    if test x"$activate_openssl" = xyes ; then
      AC_MSG_ERROR([tests for openssl failed. Change configure options])
    fi
    AC_MSG_WARN([disabling openssl support])
    activate_openssl=disabled
    activate_gsi=disabled
  else
    AC_DEFINE(HAVE_XRDCRYPTO)
    AC_DEFINE(R__SSL)
    # note: -DPERL5 is needed for resolving a clash between unistd.h and
    # openssl/des.h regarding the crypt function on some older systems (e.g. RH7.3)
    if test 0"$vernum" -lt 907 ; then
      AC_DEFINE(R__SSL_096)
      AC_SUBST(SSLCXXFLAGS,[-DPERL5])
    fi

    # note: -DR__SSL_GE_098 allows to deal with some differences in ANS1 macros
    # introduced in version 0.9.8
    if test 0"$vernum" -ge 908 ; then
      AC_DEFINE(R__SSL_GE_098)
    fi
  fi

  CPPFLAGS="$CPPFLAGS_BUP"
  LDFLAGS=$LDFLAGS_BUP
fi

AM_CONDITIONAL(ENABLE_OPENSSL,test x"$activate_openssl" = xyes -o x"$activate_openssl" = xdefault_yes)
AM_CONDITIONAL(ENABLE_GSI,test x"$activate_gsi" = xyes -o x"$activate_gsi" = xdefault_yes)

#################################################################
# Optional building of FUSE bindings for XrdFfs

ACX_LIBOPTION(fuse,[Build FUSE support for XrdFfs],yes)

if test x"$activate_fuse" = xyes -o x"$activate_fuse" = xdefault_yes ; then
  fusetests=ok

  if test x"$acx_fuse_incdir" != x; then
     searchpath=$acx_fuse_incdir
  else
     searchpath="$std_inc_path"
  fi

  ACX_LOCATEFILE([fuse.h],[$searchpath],
                 [FUSE_INCDIR="-I$acx_founddir"
                  AC_MSG_NOTICE([fuse header dir set to: $dir ]) ],
                 [if test  x"$activate_fuse" = xyes ; then
                    AC_MSG_ERROR([Could not locate fuse.h])
                  else
                    activate_fuse=disabled
                  fi
                 ]
                )

  CPPFLAGS="$FUSE_INCDIR $CPPFLAGS"
  AC_CHECK_HEADER([fuse.h],
                [],[fusetests=failed]
                )


  LDFLAGS="$FUSE_LIBDIR $LDFLAGS"
  AC_CHECK_LIB([fuse],[fuse_main],
               [:],[fusetests=failed],[-lfuse]
              )

  if test x"$fusetests" != xok ; then
    if test x"$activate_fuse" = xyes ; then
      AC_MSG_ERROR([tests for fuse failed. Change configure options])
    fi
    AC_MSG_WARN([disabling fuse support])
    activate_fuse=disabled
  fi

  CPPFLAGS="$CPPFLAGS_BUP"
  LDFLAGS="$LDFLAGS_BUP"

fi
AM_CONDITIONAL(ENABLE_FUSE,test x"$activate_fuse" = xyes -o x"$activate_fuse" = xdefault_yes)

#################################################################
# optional building of APPS
if test x"$activate_apps" = x; then
   activate_apps=default_yes
fi
AC_ARG_ENABLE(apps,
              [  --enable-apps        build XrdApps directory (xrdadler32) ],
              [],[enable_apps=$activate_apps]
             )
ACX_COLLECT_OPTION(apps,$enable_apps)
AM_CONDITIONAL(ENABLE_APPS, test x"$activate_apps" = xyes -o x"$activate_apps" = xdefault_yes)


####################################################################
# Posix I/O compatibility libraries
# There are some problems with compiling XrdPosix/XrdPss under MacOs
# Leopard (10.5). Therefore the build of these components was made
# optional
if test x"$activate_posix" = x; then
   activate_posix=default_yes
fi
if test x"$activate_apps" = xyes -o x"$activate_apps" = default_yes; then
   AC_MSG_NOTICE([Building of XrdApps (xrdadler32) requires also building of the posix library])
   activate_posix=yes
fi

AC_ARG_ENABLE(posix,
              [  --enable-posix          Enable/disable building of the posix compatibility library [[default=yes]] ],
              [],[enable_posix=$activate_posix]
             )
ACX_COLLECT_OPTION(posix,$enable_posix)

AM_CONDITIONAL(ENABLE_POSIX,test x"$activate_posix" = xyes -o x"$activate_posix" = xdefault_yes)

#################################################################
# Build XrdFfs
ACX_LIBOPTION(ffs,[Build XrdFfs], yes)
AM_CONDITIONAL(ENABLE_FFS,test x"$activate_ffs" = xyes -o x"$activate_ffs" = xdefault_yes)
# Ffs is required by Pss.
if test x"$activate_posix" = xyes -o x"$activate_posix" = xdefault_yes ; then
   if test x"$activate_ffs" = xno ; then
     AC_MSG_ERROR([posix module requires enabling of XrdFfs module - please re-run configure with ffs enabled.])
   fi
fi

####################################################################
# monitoring
if test x"$activate_mon" = x; then
   activate_mon=default_no
fi
AC_ARG_ENABLE(mon,
              [  --enable-mon            Enable/disable building of monitoring services [[default=no]] ],
              [],[enable_mon=$activate_mon]
             )
ACX_COLLECT_OPTION(mon,$enable_mon)
AM_CONDITIONAL(ENABLE_MON,test x"$activate_mon" = xyes -o x"$activate_mon" = xdefault_yes)

####################################################################
# PERL interface options
if test x"$activate_perlint" = x; then
   activate_perlint=default_yes
fi
AC_ARG_ENABLE(perlint,
              [  --enable-perlint        build perl interface [[default=yes]] ],
              [],[enable_perlint=$activate_perlint]
             )
ACX_COLLECT_OPTION(perlint,$enable_perlint)
AM_CONDITIONAL(ENABLE_PERLINT, test x"$activate_perlint" = xyes -o x"$activate_perlint" = xdefault_yes)

if test x"$activate_perlint" = xyes -o x"$activate_perlint" = xdefault_yes ; then
  AC_ARG_WITH(perl,
            [  --with-perl=PATH        path to perl executable],
            PERL=$withval,
            PERL=perl)
  AC_SUBST(PERL)
  # get the correct library options and link flags for perl
  AC_MSG_CHECKING([perl libraries and link options])
  perlbuf=`${PERL} -MExtUtils::Embed -e ldopts`
  if test x"$perlbuf" = x
  then
    AC_MSG_ERROR([no result from ${PERL} -MExtUtils::Embed -e ldopts])
  fi
  AC_MSG_RESULT($perlbuf)
  for name in $perlbuf
  do
    case $name in
      -L*)
        PERL_LIBS="${PERL_LIBS} $name"
        ;;
      -l*)
        PERL_LIBS="${PERL_LIBS} $name"
        ;;
       *.a)
        PERL_LIBS="${PERL_LIBS} $name"
        ;;
      *)
        PERL_LDOPTS="${PERL_LDOPTS} $name"
        ;;
    esac
  done
  PERL_CCOPTS=`${PERL} -MExtUtils::Embed -e ccopts`

  # since MacOS now allows packing several archs into one file and
  # this results in link errors for the unwanted arch
  # we remove all the arch flags and let the toolchain use the
  # platform defaults
  PERL_LDOPTS=`echo $PERL_LDOPTS | sed -e 's/-arch  *ppc//'`
  PERL_LDOPTS=`echo $PERL_LDOPTS | sed -e 's/-arch  *i386//'`
  PERL_LDOPTS=`echo $PERL_LDOPTS | sed -e 's/-arch  *x86_64//'`
  PERL_LDOPTS=`echo $PERL_LDOPTS | sed -e 's/-arch  *ppc64//'`
  PERL_CCOPTS=`echo $PERL_CCOPTS | sed -e 's/-arch  *ppc//'`
  PERL_CCOPTS=`echo $PERL_CCOPTS | sed -e 's/-arch  *i386//'`
  PERL_CCOPTS=`echo $PERL_CCOPTS | sed -e 's/-arch  *x86_64//'`
  PERL_CCOPTS=`echo $PERL_CCOPTS | sed -e 's/-arch  *ppc64//'`

  AC_MSG_NOTICE([perlembed libraries to link: $PERL_LIBS])
  AC_MSG_NOTICE([perlembed link flags: $PERL_LDOPTS])
  AC_MSG_NOTICE([perlembed cc options: $PERL_CCOPTS])
  AC_SUBST(PERL_LIBS)
  AC_SUBST(PERL_LDOPTS)
  AC_SUBST(PERL_CCOPTS)

  # SWIG checks for Perl interface building ####################
  if test x"$activate_swig" = x; then
     activate_swig=default_yes
  fi
  AC_ARG_ENABLE(swig,
              [  --enable-swig           enable rebuilding of Perl wrapper with SWIG [[default=yes]] ],
              [],[enable_swig=$activate_swig])
  if test $enable_swig = "yes"; then activate_swig=yes; fi
  if test $enable_swig = "no"; then activate_swig=no; fi
	
  AC_PATH_PROG(SWIG,swig,[no],[$SWIG_DIR:$PATH])
  AC_MSG_NOTICE([Using SWIG: $SWIG])
  if test x"$SWIG" = xno; then activate_swig=disabled; fi

  if test $activate_swig = "yes" -o $activate_swig = "default_yes"; then
     AC_MSG_CHECKING([for swig version])
     verstr=`$SWIG -version | grep Version 2>/dev/null`
     vermajor=`expr x"$verstr" : '.*\([[0-9]][[0-9]]*\)\.[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*'`
     verminor=`expr x"$verstr" : '.*[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\)\.[[0-9]][[0-9]]*'`
     verrelease=`expr x"$verstr" : '.*[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\)'`
     vernum=`expr 10000 \* 0"$vermajor" + 100 \* 0"$verminor" + 0"$verrelease"`
     ##AC_MSG_NOTICE([SWIG version $vermajor.$verminor.$verrelease])
     if test 0"$vernum" -lt 10333; then
	AC_MSG_RESULT([$vermajor.$verminor.$verrelease; NOT OK! need at least 1.3.33])
	activate_swig=disabled					
     else
	AC_MSG_RESULT([$vermajor.$verminor.$verrelease; OK!])
        activate_swig=yes
     fi
     AC_SUBST(SWIG)
     # if a swig rebuild is desired, we must enforce that the dependency of
     # the swig files (XrdClientAdmin_c.hh) is newer than the targets!
     chmod u+x ${srcdir}/src/XrdClient/XrdClientAdmin_c.hh
     touch ${srcdir}/src/XrdClient/XrdClientAdmin_c.hh
  fi
  ACX_COLLECT_OPTION(swig,$activate_swig)
fi

AM_CONDITIONAL(REBUILD_SWIGFILE,[test x"$activate_swig" = xyes])

#################################################################
# JAVA interface option
if test x"$activate_javaint" = x; then
   activate_javaint=default_no
fi
AC_ARG_ENABLE(javaint,
              [  --enable-javaint        build java interface [[default=no]] ],
              [],[enable_javaint=$activate_javaint])
ACX_COLLECT_OPTION(javaint,$enable_javaint)
AM_CONDITIONAL(ENABLE_JAVAINT, test x"$activate_javaint" = xyes -o x"$activate_javaint" = xdefault_yes)
MY_JAVA_HOME=$JAVA_HOME
JAVA_CLASSPATH="\$(libdir)"
AC_ARG_WITH(java-home,
	      [  --with-java-home=PATH  JAVA_HOME path],
              MY_JAVA_HOME=$withval)
AC_ARG_WITH(java-classpath,
	      [  --with-java-classpath=PATH  path where to install java .class files],
              JAVA_CLASSPATH=$withval)

if test x"$activate_javaint" = xyes -o x"$activate_javaint" = xdefault_yes; then
   if test ! -d "${MY_JAVA_HOME}/include"; then
      AC_MSG_ERROR([java include folder not found: ${MY_JAVA_HOME}/include.
   Disable javaint or use correct --with-java-home directive])
   fi
   JAVA_INCDIR="-I${MY_JAVA_HOME}/include -I${MY_JAVA_HOME}/include/linux"
   AC_MSG_NOTICE([java include settings: $JAVA_INCDIR])
   JAVA="$MY_JAVA_HOME"/bin/java
   JAVAC="$MY_JAVA_HOME"/bin/javac
   if test ! -x "$JAVA" ; then
      AC_MSG_ERROR([java executable not found at $JAVA])
   fi
   if test ! -x "$JAVAC" ; then
      AC_MSG_ERROR([java compiler not found at $JAVAC])
   fi
   javaverstr=`$JAVA -version 2>&1`
   javaverstr=`expr x"$javaverstr" : '.*\([[0-9]][[0-9]]*\.[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*[[a-z]]*\)'`
   javavermajor=`expr x"$javaverstr" : '.*\([[0-9]][[0-9]]*\).[[0-9]][[0-9]]*.[[0-9]][[0-9]]*'`
   javaverminor=`expr x"$javaverstr" : '.*[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\)\.[[0-9]][[0-9]]*'`
   javaverrelease=`expr x"$javaverstr" : '.*[[0-9]][[0-9]]*\.[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*[[a-z]]*\)'`
   javavernum=`expr 10000 \* 0"$javavermajor" + 100 \* 0"$javaverminor" + 0"$javaverrelease"`
   AC_MSG_NOTICE([java version is $javaverstr ($javavernum)])
   if test "$javavernum" -lt 10500; then
      AC_MSG_ERROR([java version > 1.5 needed])
   fi 
fi
AC_SUBST(JAVA_INCDIR)
AC_SUBST(JAVAC)
AC_SUBST(JAVA_CLASSPATH)
#################################################################
# OBSOLETE
# Castor interface options (src/XrdCS2)
# Currently disabled since building seems to require arcane knowledge
# ACX_LIBOPTION(castor,[enable building of castor interface module],no)
# activate_castor=no
# AM_CONDITIONAL(ENABLE_CASTOR, test x"$activate_castor" = xyes -o x"$activate_castor" = xdefault_yes)

#################################################################
# OBSOLETE - now built as an external plugin
# Catalog token authorization mechanism
#ACX_LIBOPTION(tkauthz,[enable building of catalog token authorization Ofs module],no)
#
#AM_CONDITIONAL(ENABLE_TOKENAUTHZ, test x"$activate_tkauthz" = xyes -o x"$activate_tkauthz" = xdefault_yes)



#################################################################
# where to install the utility scripts
AC_ARG_WITH(utils-dir,
          [  --with-utils-dir=PATH        install directory for utility scripts [[default=sysconfdir]] ],
          XROOTDUTILSDIR=$withval,
          XROOTDUTILSDIR="$bindir/xrootdutils")
AC_SUBST(XROOTDUTILSDIR)


#################################################################
# Checks for programs.

if test `expr x"$target_os" : 'xsolaris'` -gt 7 ; then
   if test -e /usr/xpg4/bin/egrep ; then
      AC_SUBST(EGREP,[/usr/xpg4/bin/egrep])
      AC_MSG_NOTICE([Solaris: using /usr/xpg4/bin/egrep for EGREP]) 
   fi
fi

dnl Disable libtool 1.5 support for languages we don't use (this is a HACK!)
define([AC_LIBTOOL_LANG_F77_CONFIG], [:])dnl
define([AC_LIBTOOL_LANG_GCJ_CONFIG], [:])dnl
# determines shared library needed for loading modules
AC_LIBTOOL_DLOPEN
# init for libtool
AC_PROG_LIBTOOL


#################################################################
# Doxygen documentation
if test x"$activate_doc" = x; then activate_doc=default_no; fi
AC_ARG_ENABLE(doc,
              [  --enable-doc            build Doxygen documentation [[default=no]] ],
              [],[enable_doc=$activate_doc])
ACX_COLLECT_OPTION(doc,$enable_doc)
AC_PATH_PROGS(DOXYGEN,[doxygen],
               [no], [$PATH])
AC_SUBST(DOXYGEN)
if test x"$DOXYGEN" = x; then
   activate_doc=disabled
fi
AM_CONDITIONAL(ENABLE_DOC, test x"$activate_doc" = xyes -o x"$activate_doc" = xdefault_yes)

####################################################################
# Test for old STL headers (<header.h>) usage 
# Rather than testing for the availability of old headers, we test for the new headers and only
# if they do not exist, switch to old header format. If these also do not exist, the error will
# show up at compile time (well... We do not need to catch everything....)
AC_CHECK_HEADER([iostream],
                [],[AC_MSG_WARN([header <iostream> not found. Using old STL header style <iostream.h>])
                    AC_DEFINE(HAVE_OLD_HDRS)]
               )

#  commented out the following code which implements the header based on a configure switch:
#if test x"$activate_oldhdrs" = x; then activate_oldhdrs=default_no; fi
#AC_ARG_ENABLE(oldhdrs,
#              [  --enable-oldhdrs        Enable/disable usage of std headers
#                          in form of <name.h> instead of <name>, e.g. <iostream.h>  [[default=disabled]]],
#              [],[enable_oldhdrs=$activate_oldhdrs]
#             )
#ACX_COLLECT_OPTION(oldhdrs,$enable_oldhdrs)
#if test x"$enable_oldhdrs" = xyes; then
#   AC_DEFINE(HAVE_OLD_HDRS)
#fi

#################################################################
# additional architecture dependent compiler flags

# this seems unnecessary and is often set in standard system includes
#AC_C_BIGENDIAN([AC_DEFINE(_BIG_ENDIAN)],
#               [AC_DEFINE(_LITTLE_ENDIAN)]
#              )

#  Note: This standard should already be included with the _GNU_SOURCE macro
AC_DEFINE(_LARGEFILE_SOURCE)

case "$target_os" in
   solaris*)
      # note: this is for the SunCC compiler
      if test x"$GCC" != xyes ; then
         CXXFLAGS="-DSUNCC -D_POSIX_PTHREAD_SEMANTICS"
         CFLAGS="-DSUNCC -D_POSIX_PTHREAD_SEMANTICS"
         if test x$target_cpu != x"i386"; then
             CXXFLAGS="$CXXFLAGS -xarch=v8plus"
             CFLAGS="$CFLAGS -xarch=v8plus"
         else
	     # workaround to a bug in swig
             PERL_CCOPTS="$PERL_CCOPTS -include sys/vnode.h"
         fi

         LDFLAGS="$LDFLAGS -mt"
         AC_SUBST(LIBSUNMATH,[-lsunmath])
      fi
      ;;
esac


#################################################################
# Checks for libraries.
AC_CHECK_LIB([pthread],[pthread_join],
             [LIBPTHREAD="-lpthread"])
AC_CHECK_LIB([pthreads],[pthread_join],
             [
             if test x"$LIBPTHREAD" != x; then
               AC_MSG_WARN([found both libpthread and libpthreads. Using -lpthread])
             else
               LIBPTHREAD="-lpthreads"
             fi
             ])
AC_SUBST(LIBPTHREAD)

AC_CHECK_LIB([rt],[aio_fsync],
             [AC_SUBST(LIBRT,[-lrt])])

AC_CHECK_LIB([z], [gzdopen],
             [AC_DEFINE(HAVE_LIBZ)
              AC_SUBST(LIBZ,[-lz])])

##  ---- DISABLED FOR NOW ----
## check for the MonALISA client cpp library
##AC_CHECK_LIB([apmoncpp],[main],
##             [:],
##             [have_apmon=xno
##              AC_MSG_WARN([libapmoncpp not found. I Will not build xrdmoncp!])
##             ])
##AM_CONDITIONAL(ENABLE_APMON, test x"$have_apmon" != xno)

#################################################################
# Checks for header files.
AC_HEADER_DIRENT
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([et/com_err.h arpa/inet.h fcntl.h inttypes.h limits.h malloc.h memory.h netdb.h netinet/in.h shadow.h stdlib.h string.h strings.h sys/file.h sys/mount.h sys/param.h sys/socket.h sys/statfs.h sys/statvfs.h sys/time.h sys/vfs.h unistd.h utime.h])

#################################################################
# Checks for typedefs, structures, and compiler characteristics.
#AC_HEADER_STDBOOL
#AC_C_CONST
#AC_TYPE_UID_T
#AC_C_INLINE
#AC_TYPE_MODE_T
#AC_TYPE_OFF_T
#AC_TYPE_PID_T
#AC_C_RESTRICT
#AC_TYPE_SIZE_T
#AC_CHECK_MEMBERS([struct stat.st_blksize])
#AC_STRUCT_ST_BLOCKS
#AC_HEADER_TIME
#AC_STRUCT_TM
#AC_C_VOLATILE
AM_PROG_CC_C_O

#################################################################
# Checks for library functions.
#AC_FUNC_CLOSEDIR_VOID
#AC_FUNC_ERROR_AT_LINE
#AC_FUNC_FORK
#AC_FUNC_LSTAT
#AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
#AC_FUNC_MALLOC
#AC_FUNC_MEMCMP
#AC_FUNC_MKTIME
#AC_FUNC_MMAP
#AC_FUNC_REALLOC
#AC_TYPE_SIGNAL
#AC_FUNC_STAT
#AC_FUNC_STRERROR_R
#AC_FUNC_UTIME_NULL
#AC_FUNC_VPRINTF
#AC_CHECK_FUNCS([bzero dup2 endgrent ftruncate getcwd gethostbyaddr gethostbyname gethostname getpagesize getpass getspnam gettimeofday inet_ntoa localtime_r memchr memmove memset mkdir munmap pathconf putenv rmdir socket strcasecmp strchr strcspn strdup strerror strncasecmp strspn strstr strtol utime])

AC_CHECK_FUNC([getnameinfo],[AC_DEFINE(HAVE_NAMEINFO)],
              [AC_CHECK_LIB([socket],[getnameinfo],
                 [
                 AC_DEFINE(HAVE_NAMEINFO)
                 AC_SUBST(LIBSOCKET,[-lsocket])
                 ])
              ])

AC_CHECK_FUNC([getprotobyname_r],[AC_DEFINE(HAVE_PROTOR)],
              [AC_CHECK_LIB([socket],[getprotobyname_r],
                  [
                  AC_DEFINE(HAVE_PROTOR)
                  AC_SUBST(LIBSOCKET,[-lsocket])
                  ])
              ])

AC_CHECK_FUNC([strlcpy],[AC_DEFINE(HAVE_STRLCPY)])
AC_CHECK_FUNC([sendfile],[AC_DEFINE(HAVE_SENDFILE)])

AC_CHECK_FUNC([sendfile],[AC_DEFINE(HAVE_SENDFILE)])

AC_CHECK_FUNCS([fstatat])

#   note: needed for solaris
AC_CHECK_LIB([rt],[sem_post],
             [AC_SUBST(SEM_LIBRT,[-lrt])]
            )

#   note: needed for solaris (network services library. many systems still have these
#         functions still in the libc)
AC_CHECK_LIB([nsl],[gethostbyname],
             [AC_SUBST(LIBNSL,[-lnsl])]
            )

AC_CHECK_FUNC(sigwaitinfo,[AC_DEFINE(HAVE_SIGWTI)],
                [AC_CHECK_LIB([rt],[sigwaitinfo],
                   [
                   AC_DEFINE(HAVE_SIGWTI)
                   AC_SUBST(LIBRT,[-lrt])
                   ])
                ])

# On linux we need to check if setresuid prototype is defined
case "$target_os" in
   linux*)
      AC_CHECK_FUNCS(setresuid,
                     AC_DEFINE(HAVE_SETRESUID)
                    )
      ;;
esac

AX_HAVE_EPOLL(AC_DEFINE(_EPOLL),[])
AC_SYS_DEV_POLL(AC_DEFINE(_DEVPOLL),[])

###########################################
# GNU Readline checks
ACX_LIBOPTION(readline,[use GNU Readline support],yes)

# extend the preprocessor and link PATHs to the Vars provided by ACX_LIBOPTION
CPPFLAGS="$READLINE_INCDIR $CPPFLAGS"
LDFLAGS="$READLINE_LIBDIR $LDFLAGS"

if test x"$activate_readline" = xyes -o x"$activate_readline" = xdefault_yes ; then
   AC_CHECK_HEADERS([readline/readline.h readline/history.h curses.h term.h],
                  [],[activate_readline="no"]
		 )
fi

if test x"$activate_readline" = xyes -o x"$activate_readline" = xdefault_yes ; then
   # Note: some systems like Mac offer all readline symbols within libreadline and
   # have no additional libhisory installed on the system
   AC_CHECK_LIB([history],[clear_history],
               [HISTORY_LIB="-lhistory"]
               ,,[-lcurses])
   AC_CHECK_LIB([readline],[readline],
               [AC_DEFINE(HAVE_READLINE)
               READLINE_LIBS="-lreadline $HISTORY_LIB -lcurses"]
               ,[activate_readline="no"],[-lcurses])
fi
AC_SUBST(READLINE_LIBS)

if test x"$activate_readline" != xyes -a  x"$activate_readline" != xdefault_yes ; then
   AC_MSG_WARN([xrd application will be built without GNU readline support])
fi

CPPFLAGS="$CPPFLAGS_BUP"
LDFLAGS="$LDFLAGS_BUP"

###########################################
# Kerberos 4 checks
krb4tests=ok
CPPFLAGS="$KRB4_INCDIR $CPPFLAGS"
AC_CHECK_HEADER([kerberosIV/krb.h],
                [],[krb4tests=failed]
               )

LDFLAGS="$KRB4_LIBDIR $LDFLAGS"
AC_CHECK_LIB([krb4],[krb_get_lrealm],
             [:],[krb4tests=failed],[-ldes425 -lkrb5 -lk5crypto -lcom_err -lresolv  $LIBSOCKET  $LIBNSL]
            )

if test x"$activate_krb4" = xyes -o x"$activate_krb4" = xdefault_yes; then
  if test x"$krb4tests" = xfailed ; then
    if test x"$enable_krb4" = xyes ; then
      ACX_MSG_ERROR([tests for krb4 failed. Change configure options!])
    fi
      AC_MSG_WARN([tests for krb4 failed. Disabling krb4 support])
    activate_krb4=disabled
  fi
fi
CPPFLAGS="$CPPFLAGS_BUP"
LDFLAGS="$LDFLAGS_BUP"

AM_CONDITIONAL(ENABLE_KRB4,test x"$activate_krb4" = xyes -o x"$activate_krb4" = xdefault_yes)

###########################################
# Kerberos 5 checks
krb5tests=ok
CPPFLAGS="$KRB5_INCDIR $CPPFLAGS"
AC_CHECK_HEADER([krb5.h],
                [],[krb5tests=failed]
               )

LDFLAGS="$KRB5_LIBDIR $LDFLAGS"
AC_CHECK_LIB([krb5],[krb5_init_context],
             [:],[krb5tests=failed],[-lk5crypto -lcom_err]
            )
if test x"$activate_krb5" = xyes -o x"$activate_krb5" = xdefault_yes; then
  if test x"$krb5tests" = xfailed ; then
    if test x"$enable_krb5" = xyes ; then
      ACX_MSG_ERROR([tests for krb5 failed. Change configure options!])
    fi
    AC_MSG_WARN([tests for krb5 failed. Disabling krb5 support])
    activate_krb5=disabled
  fi
fi

LDFLAGS="$LDFLAGS_BUP"
CPPFLAGS="$CPPFLAGS_BUP"

AM_CONDITIONAL(ENABLE_KRB5,test x"$activate_krb5" = xyes -o x"$activate_krb5" = xdefault_yes)


###########################################
# libxml2 checks
ACX_LIBOPTION(secssl,[SecSSL support], yes)
ACX_LIBOPTION(xml2,[xml library],yes)

if test x"$activate_xml2" = xyes -o x"$activate_xml2" = xdefault_yes; then
 if test x"$acx_xml2_incdir" != x; then
   searchpath=$acx_xml2_incdir
 else
   searchpath="/usr/include/libxml2"
  fi

  ACX_LOCATEFILE([libxml/xmlmemory.h],[$searchpath],
                 [XML2_INCDIR="-I$acx_founddir"
                  AC_MSG_NOTICE([xml2 header dir set to: $dir ]) ],
                 [activate_xml2=no]
                )

  CPPFLAGS_BUP="$CPPFLAGS"
  LDFLAGS_BUP=$LDFLAGS

  CPPFLAGS="$XML2_INCDIR $CPPFLAGS"

  AC_CHECK_HEADER([libxml/xmlmemory.h],
                  [],[activate_xml2=no]
                  )

  LDFLAGS="$XML2_LIBDIR $LDFLAGS"
  AC_CHECK_LIB([xml2],[xmlReaderForMemory],
               [:],[activate_xml2=no],[]
              )

  CPPFLAGS="$CPPFLAGS_BUP"
  LDFLAGS=$LDFLAGS_BUP

  AC_SUBST(XML2_INCDIR)
  AC_SUBST(XML2_LIBDIR)
fi

if test x"$activate_secssl" = xyes -o x"$activate_secssl" = xdefault_yes; then
   if test x"$ssltests" != xok; then
     activate_secssl=no
     AC_MSG_WARN([Cannot build SecSSL plugin without SSL support])
   fi
   if test x"$activate_xml2" = xno; then
     activate_secssl=no
     AC_MSG_WARN([Cannot build SecSSL plugin without XML2 support])
   fi
fi

AM_CONDITIONAL(ENABLE_SECSSL,test x"$activate_secssl" = xyes -o x"$activate_secssl" = xdefault_yes)

# Record the extension of dynamic libraries on this platform
LT_SYS_MODULE_EXT

###########################################
# tkauthz checks
# currently disabled since they would require the AC_LANG(C++) macro
# (see at the beginning of this file)


#if test x"$activate_tkauthz" = xyes -o x"$activate_tkauthz" = xdefault_yes; then
#   CPPFLAGS="$TKAUTHZ_INCDIR $CPPFLAGS"
#   AC_CHECK_HEADER([TTokenAuthz.h],
#                [],[tkauthztests=failed]
#            )
#   if test x"$tkauthztests" = xfailed; then
#      if test x"$activate_tkauthz" = xyes; then
#         AC_MSG_ERROR([tkauthz tests failed])
#      else
#         AC_MSG_WARN([tkauthz tests failed - disabling])
#         activate_tkauthz=disabled
#      fi
#   fi
#
#   CPPFLAGS="$CPPFLAGS_BUP"
#fi

#################################################################


CXXFLAGS="$CXXFLAGS -D_REENTRANT"


AC_OUTPUT(Makefile
          src/Makefile
          src/Makefile_include
          src/XrdSys/Makefile
          src/XrdOuc/Makefile
          src/XrdNet/Makefile
          src/XrdSec/Makefile
          src/XrdSut/Makefile
	  src/XrdCrypto/Makefile
          src/XrdSecpwd/Makefile
	  src/XrdSeckrb4/Makefile
	  src/XrdSeckrb5/Makefile
	  src/XrdSecgsi/Makefile
          src/XrdSecsss/Makefile
	  src/XrdSecssl/Makefile
	  src/XrdSecssl/libsslGridSite/Makefile
	  src/XrdSecunix/Makefile
	  src/XrdSfs/Makefile
          src/XrdAcc/Makefile
          src/XrdBwm/Makefile
          src/XrdOss/Makefile
          src/XrdOfs/Makefile
          src/Xrd/Makefile
          src/XrdCms/Makefile
          src/XrdRootd/Makefile
          src/XrdXrootd/Makefile
          src/XrdFrm/Makefile
          src/XrdClient/Makefile
          src/XrdPosix/Makefile
          src/XrdPss/Makefile
          src/XrdMon/Makefile
          src/XrdCns/Makefile
          src/XrdApps/Makefile
          src/XrdFfs/Makefile
          test/testconfig.sh)


#####################################################################
# we need to move the GNUMakefiles of the classic build away
if test -e src/XrdOuc/GNUmakefile ; then
   echo "###########################################################"
   echo "Moving away the old GNUmakefiles to GNUmakefile.classic"
   if test `expr x"$target_os" : 'xsolaris'` -gt 7 ; then
      find . -name GNUmakefile | xargs -i mv {} {}.classic 
   else
      find . -name GNUmakefile -exec mv {} {}.classic \;
   fi
fi


ACX_PRINTOPTIONS
