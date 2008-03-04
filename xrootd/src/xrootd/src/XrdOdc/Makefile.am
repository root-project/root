#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.4 2007/06/07 10:44:41 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdOdc.la

## Currently commented out. Compilation of the test will yield an error
## bin_PROGRAMS = XrdOdcTest

xrootdodcincdir = $(includedir)/xrootd/XrdOdc

xrootdodcinc_HEADERS = \
    XrdOdcConfDefs.hh  XrdOdcConfig.hh  XrdOdcFinder.hh \
    XrdOdcManager.hh  XrdOdcMsg.hh  XrdOdcResp.hh \
    XrdOdcTrace.hh

INCLUDES = -I$(top_srcdir)/src

libXrdOdc_la_SOURCES = \
        XrdOdcConfig.cc XrdOdcConfig.hh XrdOdcConfDefs.hh \
        XrdOdcFinder.cc XrdOdcFinder.hh \
        XrdOdcManager.cc XrdOdcManager.hh \
        XrdOdcMsg.cc XrdOdcMsg.hh XrdOdcTrace.hh \
        XrdOdcResp.cc XrdOdcResp.hh

## XrdOdcTest_SOURCES = XrdOdcTest.cc
## XrdOdcTest_LDADD = libXrdOdc.la \
##                   ../XrdNet/libXrdNet.la \
##                   ../XrdOuc/libXrdOuc.la

