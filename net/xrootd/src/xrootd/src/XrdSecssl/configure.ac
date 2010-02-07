AC_PREREQ(2.57)
AC_INIT(xrootd-secssl, 4.0.1,[andreas.joachim.peters@cern.ch])
AC_CONFIG_SRCDIR([XrdSecProtocolsslTrace.hh])
AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE(1.6 foreign)

mkdir -p m4
AC_CONFIG_MACRO_DIR([m4])

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

libversion=`libtoolize --version | head -1  | cut -d " " -f 4`
case "$target_os" in
   darwin*)
   if test "x$libversion" = "x1.5.22"; then
      AC_MSG_ERROR([broken libtool version $libversion found on this MAC - install newer one from http://www.gnu.org/software/libtool/news
.html])
   fi
   ;;
esac


echo $libdir | grep lib64 >& /dev/null
if test "$?" = "0" ; then 
   MARK64=64;
else
   MARK64="";
fi

AC_SUBST(MARK64)

AC_DEFINE(_LARGEFILE_SOURCE)
AC_DEFINE(_FILE_OFFSET_BITS,64)
AC_SYS_LARGEFILE

# Do all testing using C++
AC_LANG([C++])

# Checks for programs.
AC_PROG_CXX
AC_PROG_LIBTOOL
LT_INIT

AC_ARG_WITH(xrootd-location,
          [  --with-xrootd-location=installation of xrootd version to use],
          [XROOTD_LOCATION=$withval],
          [XROOTD_LOCATION=/opt/xrootd/])

AC_ARG_WITH(xrootd-incdir,
          [  --with-xrootd-incdir=installation of xrootd include files],
          [XROOTD_INCDIR=$withval],
          [XROOTD_INCDIR=$XROOTD_LOCATION/include/xrootd/])

AC_ARG_WITH(xrootd-libdir,
          [  --with-xrootd-libdir=installation of xrootd libraries],
          [XROOTD_LIBDIR=$withval],
          [XROOTD_LIBDIR=$XROOTD_LOCATION/lib])

AC_SUBST(XROOTD_INCDIR)
AC_SUBST(XROOTD_LIBDIR)

if ! test -e "${XROOTD_INCDIR}/XrdOuc/XrdOucEnv.hh"
then
   AC_MSG_ERROR([xrootd header files could not be found in inc directory ${XROOTD_INCDIR}])
fi

# Check for xrootd library
#if ! test -e "${XROOTD_LIBDIR}/libXrdOuc.so"
#then
#   AC_MSG_ERROR([xrootd library could not be found in lib directory ${XROOTD_LIBDIR}])
#fi

SSLLINKLIB="-L${XROOTD_LIBDIR}/ -lXrdOuc"

####################################################################
# OpenSSL support
  ACX_LIBOPTION(openssl,[ssl library],yes)
  ssltests=ok

  if test x"$acx_openssl_incdir" != x; then
     searchpath=$acx_openssl_incdir
  else
     searchpath="/usr/include"
  fi

  ACX_LOCATEFILE([openssl/opensslv.h],[$searchpath],
                 [OPENSSL_INCDIR="-I$acx_founddir"
                  OPENSSL_VERSION_TEXT=`grep OPENSSL_VERSION_TEXT "$dir"/openssl/opensslv.h | grep OpenSSL`
                  AC_MSG_NOTICE([openssl header dir set to: $dir ]) ],
                 [AC_MSG_ERROR([Could not locate openssl/opensslv.h])]
                )

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

  CPPFLAGS_BUP="$CPPFLAGS"
  LDFLAGS_BUP=$LDFLAGS

  CPPFLAGS="$OPENSSL_INCDIR $CPPFLAGS"
  AC_CHECK_HEADER([openssl/ssl.h],
                [],[ssltests=failed]
                )


  LDFLAGS="$OPENSSL_LIBDIR $LDFLAGS"
  AC_CHECK_LIB([ssl],[SSL_library_init],
               [:],[ssltests=failed],[-lcrypto]
              )

  if test x"$ssltests" != xok ; then
    AC_MSG_ERROR([tests for openssl failed. Change configure options])
  else
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

####################################################################
# libxml2 support
  xmltests=ok
  ACX_LIBOPTION(xml2,[xml library],yes)
  if test x"$acx_xml2_incdir" != x; then
     searchpath=$acx_xml2_incdir
  else
     searchpath="/usr/include/libxml2"
  fi

  ACX_LOCATEFILE([libxml/xmlmemory.h],[$searchpath],
                 [XML2_INCDIR="-I$acx_founddir"
                  AC_MSG_NOTICE([xml2 header dir set to: $dir ]) ],
                 [AC_MSG_ERROR([Could not locate libxml/xmlmemory.h])]
                )

  CPPFLAGS_BUP="$CPPFLAGS"
  LDFLAGS_BUP=$LDFLAGS

  CPPFLAGS="$XML2_INCDIR $CPPFLAGS"

  AC_CHECK_HEADER([libxml/xmlmemory.h],
                [],[ssltests=failed]
                )


  LDFLAGS="$XML2_LIBDIR $LDFLAGS"
  AC_CHECK_LIB([xml2],[xmlReaderForMemory],
               [:],[xmltests=failed],[]
              )

  if test x"$xmltests" != xok ; then
    AC_MSG_ERROR([tests for libxml2 failed. Change configure options])
  else
    AC_MSG_NOTICE([using xml settings: $XML2_INCDIR $XML2_LIBDIR ])
  fi

  CPPFLAGS="$CPPFLAGS_BUP"
  LDFLAGS=$LDFLAGS_BUP

AC_SUBST(XML2_INCDIR)
AC_SUBST(XML2_LIBDIR)

AC_SUBST(IS_XRDSECSSL)
AC_SUBST(SSLLINKLIB)
# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([sys/time.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_MODE_T
AC_TYPE_SIZE_T
AC_HEADER_TIME

# Checks for library functions.
AC_FUNC_STAT
AC_CHECK_FUNCS([gettimeofday mkdir strdup])

AC_CONFIG_FILES([Makefile
	libsslGridSite/Makefile 
        xrootd-secssl.spec  
] )

test -L XrdSecssl || ln -s ./ XrdSecssl
echo "=================================================="
echo "Configuring for ../lib$MARK64 library directories"
echo "=================================================="
AC_OUTPUT
