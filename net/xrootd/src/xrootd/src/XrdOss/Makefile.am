#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.4 2008/03/24 06:30:12 abh Exp $
## Checked in by $Author: abh $
#######################################################################

noinst_HEADERS = XrdOss.hh

lib_LTLIBRARIES = libXrdOss.la

INCLUDES = -I$(top_srcdir)/src

xrootdossincdir = $(includedir)/xrootd/XrdOss

xrootdossinc_HEADERS = \
    XrdOss.hh        XrdOssError.hh    XrdOssOpaque.hh \
    XrdOssApi.hh     XrdOssLock.hh     XrdOssTrace.hh \
    XrdOssCache.hh   XrdOssMio.hh      XrdOssPath.hh \
    XrdOssSpace.hh   XrdOssStage.hh \
    XrdOssConfig.hh  XrdOssMioFile.hh


libXrdOss_la_SOURCES = \
        XrdOssAio.cc XrdOssTrace.hh \
        XrdOssApi.cc XrdOssApi.hh XrdOssError.hh \
        XrdOssCache.cc XrdOssCache.hh \
        XrdOssConfig.cc XrdOssConfig.hh \
        XrdOssCreate.cc XrdOssOpaque.hh \
        XrdOssLock.cc XrdOssLock.hh   \
        XrdOssMio.cc XrdOssMio.hh XrdOssMioFile.hh   \
        XrdOssMSS.cc    \
        XrdOssPath.cc XrdOssPath.hh \
        XrdOssRename.cc \
        XrdOssStage.cc  \
        XrdOssSpace.cc XrdOssSpace.hh \
        XrdOssStat.cc \
        XrdOssUnlink.cc
libXrdOss_la_LIBADD = $(LIBRT)
