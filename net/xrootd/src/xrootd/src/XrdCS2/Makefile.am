#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.1 2007/07/17 07:04:55 ganis Exp $
## Checked in by $Author: ganis $
#######################################################################

lib_LTLIBRARIES = libXrdCS2.la

xrootdcs2incdir = $(includedir)/xrootd/XrdCS2

xrootdcs2inc_HEADERS = XrdCS2DCM.hh XrdCS2DCMFile.hh \
		       XrdCS2Req.hh XrdCS2Xmi.hh

INCLUDES = -I$(top_srcdir)/src $(CASTOR_INCDIR)

libXrdCS2_la_SOURCES = \
        XrdCS2DCM.cc XrdCS2DCMConfig.cc XrdCS2DCMMain.cc XrdCS2Xmi.cc \
        XrdCS2DCM.hh XrdCS2DCMFile.cc XrdCS2Req.cc XrdCS2Xmi.hh \
        XrdCS2DCM2cs.cc XrdCS2DCMFile.hh XrdCS2Req.hh

libXrdCS2_la_LIBADD = $(CASTOR_LIBDIR) -lshift

