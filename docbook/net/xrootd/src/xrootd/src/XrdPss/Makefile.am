#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

lib_LTLIBRARIES = libXrdPss.la

xrootdpssincdir = $(includedir)/xrootd/XrdPss
xrootdpssinc_HEADERS = XrdPss.hh

INCLUDES = -I$(top_srcdir)/src -I$(top_builddir)/src

libXrdPss_la_SOURCES = \
        XrdPss.cc  XrdPss.hh \
        XrdPssAio.cc \
        XrdPssConfig.cc

libXrdPss_la_LIBADD =   ../XrdFfs/libXrdFfs.la  \
                        ../XrdClient/libXrdClient.la \
                        ../XrdSecsss//libXrdSecsss.la \
                        ../XrdSec//libXrdSec.la \
                        ../XrdNet/libXrdNet.la \
                        ../XrdNet/libXrdNetUtil.la \
                        ../XrdOuc/libXrdOuc.la \
                        ../XrdSys/libXrdSys.la
