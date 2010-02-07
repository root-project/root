#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.5 2009/05/22 01:04:24 abh Exp $
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
    XrdOssCopy.hh    XrdOssSpace.hh    XrdOssStage.hh \
    XrdOssConfig.hh  XrdOssMioFile.hh


libXrdOss_la_SOURCES = \
        XrdOssAio.cc XrdOssTrace.hh \
        XrdOssApi.cc XrdOssApi.hh XrdOssError.hh \
        XrdOssCache.cc XrdOssCache.hh \
        XrdOssConfig.cc XrdOssConfig.hh \
        XrdOssCopy.cc XrdOssCopy.hh \
        XrdOssCreate.cc XrdOssOpaque.hh \
        XrdOssLock.cc XrdOssLock.hh   \
        XrdOssMio.cc XrdOssMio.hh XrdOssMioFile.hh   \
        XrdOssMSS.cc    \
        XrdOssPath.cc XrdOssPath.hh \
        XrdOssReloc.cc \
        XrdOssRename.cc \
        XrdOssStage.cc  \
        XrdOssSpace.cc XrdOssSpace.hh \
        XrdOssStat.cc \
        XrdOssUnlink.cc
libXrdOss_la_LIBADD = $(LIBRT)
