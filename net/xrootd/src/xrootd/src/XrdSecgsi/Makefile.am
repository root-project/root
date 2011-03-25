#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

lib_LTLIBRARIES = libXrdSecgsi.la libXrdSecgsiGMAPLADP.la

bin_PROGRAMS = xrdgsiproxy
man1_MANS = xrdgsiproxy.1

# TODO: xrdsecgsitest no longer builds
#check_PROGRAMS = xrdsecgsitest

INCLUDES = -I$(top_srcdir)/src

libXrdSecgsi_la_SOURCES = \
        XrdSecProtocolgsi.cc XrdSecProtocolgsi.hh XrdSecgsiTrace.hh
libXrdSecgsi_la_LDFLAGS = -module

libXrdSecgsi_la_LIBADD = ../XrdCrypto/libXrdCryptossl.la \
                         ../XrdCrypto/libXrdCrypto.la \
                         ../XrdSec/libXrdSec.la \
                         ../XrdOuc/libXrdOuc.la \
                         ../XrdNet/libXrdNet.la \
                         ../XrdSys/libXrdSys.la
# following line disabled. Look for reason in XrdSec/Makefile.am
#libXrdSecgsi_la_LIBTOOLFLAGS = --tag=disable-static

# GridMAP function looking up an LDAP database
libXrdSecgsiGMAPLADP_la_SOURCES = XrdSecgsiGMAPFunLDAP.cc
libXrdSecgsiGMAPLADP_la_LDFLAGS = -module

xrdgsiproxy_SOURCES = XrdSecgsiProxy.cc
xrdgsiproxy_LDFLAGS =  -static
xrdgsiproxy_LDADD = libXrdSecgsi.la \
                    ../XrdSec/libXrdSec.la \
                    ../XrdOuc/libXrdOuc.la \
                    ../XrdNet/libXrdNet.la \
                    ../XrdSys/libXrdSys.la


#xrdsecgsitest_SOURCES = XrdSecgsitest.cc
#xrdsecgsitest_LDADD = libXrdSecgsi.la \
#                     ../XrdSec/libXrdSec.la \
#                     ../XrdOuc/libXrdOuc.la \
#                     ../XrdNet/libXrdNet.la

