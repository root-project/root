#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.2 2007/06/01 10:29:53 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdProxy.la

xrootdpssincdir = $(includedir)/xrootd/XrdPss
xrootdpssinc_HEADERS = XrdPss.hh

INCLUDES = -I$(top_srcdir)/src

libXrdProxy_la_SOURCES = \
        XrdPss.cc  XrdPss.hh \
        XrdPssAio.cc \
        XrdPssConfig.cc

libXrdProxy_la_LIBADD = ../XrdClient/libXrdClient.la \
                        ../XrdNet/libXrdNet.la \
                        ../XrdOuc/libXrdOuc.la \
                        ../XrdSys/libXrdSys.la
