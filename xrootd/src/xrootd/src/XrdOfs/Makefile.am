#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.9 2007/08/09 17:02:20 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdOfs.la

INCLUDES = -I$(top_srcdir)/src

xrootdofsincdir = $(includedir)/xrootd/XrdOfs

xrootdofsinc_HEADERS = \
    XrdOfs.hh  XrdOfs.icc  XrdOfsConfig.hh \
    XrdOfsEvr.hh  XrdOfsEvs.hh  XrdOfsHandle.hh \
    XrdOfsSecurity.hh  XrdOfsTrace.hh \
    XrdOfsConfig.cc XrdOfsEvs.cc XrdOfsEvr.cc XrdOfsHandle.cc XrdOfs.cc
# Note: the source files above also get installed, so that third party
# derived objects of the Ofs can be built

libXrdOfs_la_SOURCES = \
        XrdOfs.cc XrdOfs.hh XrdOfsSecurity.hh XrdOfsTrace.hh XrdOfs.icc \
        XrdOfsConfig.cc XrdOfsConfig.hh \
        XrdOfsEvr.cc XrdOfsEvr.hh   \
        XrdOfsEvs.cc XrdOfsEvs.hh   \
        XrdOfsHandle.cc XrdOfsHandle.hh

libXrdOfs_la_LIBTOOLFLAGS = --tag=disable-static
libXrdOfs_la_LDFLAGS = -module

libXrdOfs_la_LIBADD = ../XrdAcc/libXrdAcc.la \
                      ../XrdOss/libXrdOss.la \
                      ../XrdOdc/libXrdOdc.la \
                      ../XrdNet/libXrdNet.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdSys/libXrdSys.la

