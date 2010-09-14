#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.7 2010/08/05 15:25:01 feichtinger Exp $
## Checked in by $Author: feichtinger $
#######################################################################

AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra directory definitions for install area

xrootdaccincdir = $(includedir)/xrootd/XrdAcc

#############################################################################
## deliverables

lib_LTLIBRARIES = libXrdAcc.la

bin_PROGRAMS = XrdAccTest


xrootdaccinc_HEADERS = \
    XrdAccAccess.hh  XrdAccAuthFile.hh    XrdAccConfig.hh \
    XrdAccAudit.hh   XrdAccAuthorize.hh   XrdAccGroups.hh \
    XrdAccAuthDB.hh  XrdAccCapability.hh  XrdAccPrivs.hh

#############################################################################
## Build rules

libXrdAcc_la_SOURCES = \
        XrdAccAccess.cc XrdAccAccess.hh     \
        XrdAccAudit.cc XrdAccAudit.hh      \
        XrdAccAuthFile.cc XrdAccAuthFile.hh   \
        XrdAccCapability.cc XrdAccCapability.hh \
        XrdAccConfig.cc XrdAccConfig.hh XrdAccAuthDB.hh    \
        XrdAccGroups.cc XrdAccGroups.hh XrdAccPrivs.hh

XrdAccTest_SOURCES = XrdAccTest.cc XrdAccAuthorize.hh
XrdAccTest_LDADD = libXrdAcc.la \
                   ../XrdOuc/libXrdOuc.la \
                   ../XrdNet/libXrdNet.la \
                   ../XrdSys/libXrdSys.la \
                   ../XrdNet/libXrdNetUtil.la \
                   -ldl
