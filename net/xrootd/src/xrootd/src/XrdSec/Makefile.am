#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.7 2008/12/11 02:48:46 abh Exp $
## Checked in by $Author: abh $
#######################################################################
xrootdsecincdir = $(includedir)/xrootd/XrdSec

xrootdsecinc_HEADERS = XrdSecInterface.hh XrdSecEntity.hh

lib_LTLIBRARIES = libXrdSec.la

bin_PROGRAMS = testclient testserver

INCLUDES = -I$(top_srcdir)/src

libXrdSec_la_SOURCES = \
        XrdSecClient.cc XrdSecInterface.hh XrdSecEntity.hh      \
        XrdSecPManager.cc XrdSecPManager.hh     \
        XrdSecProtocolhost.cc XrdSecProtocolhost.hh \
        XrdSecServer.cc XrdSecServer.hh \
        XrdSecTLayer.cc XrdSecTLayer.hh \
        XrdSecTrace.hh
libXrdSec_la_LDFLAGS = -module

# Note: the next option leads to problems at least on Darwin
#libXrdSec_la_LIBTOOLFLAGS = --tag=disable-static
#  loadable modules are not equal to shared libraries, even though
#  this holds true for many systems
#  So, the testclient linking is actually not portable. Libtool can go
#  around this problem, if a static library for libXrdSec is available.
#  The clean way would be to either build a module and a library, or
#  make the testclient load the loadable module.

testclient_SOURCES = XrdSectestClient.cc
testclient_LDFLAGS = -static
testclient_LDADD =   libXrdSec.la \
                     -ldl \
                     ../XrdOuc/libXrdOuc.la \
                     ../XrdNet/libXrdNet.la \
                     ../XrdSys/libXrdSys.la

testserver_SOURCES = XrdSectestServer.cc
testserver_LDFLAGS = -static
testserver_LDADD = libXrdSec.la \
                   -ldl \
                   ../XrdOuc/libXrdOuc.la \
                   ../XrdNet/libXrdNet.la \
                   ../XrdSys/libXrdSys.la
