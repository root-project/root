#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.3 2007/06/07 10:44:48 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

noinst_HEADERS = XrdOss.hh

lib_LTLIBRARIES = libXrdOss.la

INCLUDES = -I$(top_srcdir)/src

xrootdossincdir = $(includedir)/xrootd/XrdOss

xrootdossinc_HEADERS = \
    XrdOss.hh        XrdOssError.hh    XrdOssOpaque.hh \
    XrdOssApi.hh     XrdOssLock.hh     XrdOssTrace.hh \
    XrdOssCache.hh   XrdOssMio.hh \
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
        XrdOssRename.cc \
        XrdOssStage.cc  \
        XrdOssUnlink.cc
libXrdOss_la_LIBADD = $(LIBRT)
