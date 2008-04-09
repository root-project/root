#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.6 2007/06/07 10:45:03 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdSecpwd.la

bin_PROGRAMS = xrdpwdadmin

INCLUDES = -I$(top_srcdir)/src

libXrdSecpwd_la_SOURCES = \
        XrdSecProtocolpwd.cc XrdSecProtocolpwd.hh XrdSecpwdPlatform.hh
libXrdSecpwd_la_LIBADD = ../XrdCrypto/libXrdCrypto.la \
                      ../XrdSut/libXrdSut.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdSys/libXrdSys.la \
		      $(CRYPTLIB) $(OPENSSL_LIBDIR) -ldl -lcrypto
libXrdSecpwd_la_LDFLAGS = -module
libXrdSecpwd_la_LIBTOOLFLAGS = --tag=disable-static

xrdpwdadmin_SOURCES = XrdSecpwdSrvAdmin.cc
xrdpwdadmin_LDFLAGS = -static
xrdpwdadmin_LDADD =   ../XrdCrypto/libXrdCrypto.la \
                      ../XrdSut/libXrdSut.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdSys/libXrdSys.la \
		      $(CRYPTLIB) $(OPENSSL_LIBDIR) -ldl -lcrypto

