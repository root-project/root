#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.6 2007/08/02 15:35:23 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdSecgsi.la

bin_PROGRAMS = xrdgsiproxy

# TODO: xrdsecgsitest no longer builds
#check_PROGRAMS = xrdsecgsitest

INCLUDES = -I$(top_srcdir)/src

libXrdSecgsi_la_SOURCES = \
        XrdSecProtocolgsi.cc XrdSecProtocolgsi.hh XrdSecgsiTrace.hh
libXrdSecgsi_la_LDFLAGS = -module
libXrdSecgsi_la_LIBADD = ../XrdCrypto/libXrdCrypto.la
# following line disabled. Look for reason in XrdSec/Makefile.am
#libXrdSecgsi_la_LIBTOOLFLAGS = --tag=disable-static

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

