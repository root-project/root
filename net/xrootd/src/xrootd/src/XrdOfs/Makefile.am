#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.15 2009/07/15 01:56:34 abh Exp $
## Checked in by $Author: abh $
#######################################################################

lib_LTLIBRARIES = libXrdOfs.la

INCLUDES = -I$(top_srcdir)/src

xrootdofsincdir = $(includedir)/xrootd/XrdOfs

xrootdofsinc_HEADERS = \
    XrdOfs.hh \
    XrdOfsEvr.hh  XrdOfsEvs.hh  XrdOfsHandle.hh \
    XrdOfsSecurity.hh  XrdOfsTrace.hh  XrdOfsConfig.hh \
    XrdOfsConfig.cc XrdOfsEvs.cc XrdOfsEvr.cc XrdOfsHandle.cc XrdOfs.cc \
    XrdOfsPoscq.hh XrdOfsStats.hh
# Note: the source files above also get installed in the headers directory,
# so that derived objects of the Ofs can be built. This was originally
# needed for the ALICE tokenauthz extension.

libXrdOfs_la_SOURCES = \
        XrdOfs.cc XrdOfs.hh XrdOfsSecurity.hh XrdOfsTrace.hh XrdOfsFS.cc \
        XrdOfsConfig.cc \
        XrdOfsEvr.cc XrdOfsEvr.hh   \
        XrdOfsEvs.cc XrdOfsEvs.hh   \
        XrdOfsHandle.cc XrdOfsHandle.hh \
        XrdOfsPoscq.cc  XrdOfsStats.cc

libXrdOfs_la_LIBTOOLFLAGS = --tag=disable-static
libXrdOfs_la_LDFLAGS = -module

libXrdOfs_la_LIBADD = ../XrdAcc/libXrdAcc.la \
                      ../XrdOss/libXrdOss.la \
                      ../XrdCms/libXrdCms.la \
                      ../XrdOdc/libXrdOdc.la \
                      ../XrdNet/libXrdNet.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdSys/libXrdSys.la

