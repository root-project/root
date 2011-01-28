#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 17.5.2008
##
## Version info: $Id$
## Checked in by $Author: dfeich $
#######################################################################

#############################################################################
## extra directory definitions
xrootdbwmincdir = $(includedir)/xrootd/XrdBwm

#############################################################################
## deliverables
lib_LTLIBRARIES = libXrdBwm.la 

xrootdbwminc_HEADERS = XrdBwm.hh \
                       XrdBwmHandle.hh \
                       XrdBwmLogger.hh \
                       XrdBwmPolicy.hh \
                       XrdBwmPolicy1.hh \
                       XrdBwmTrace.hh

#############################################################################
## Build rules
AM_CPPFLAGS = -I$(top_srcdir)/src

libXrdBwm_la_SOURCES = \
                       XrdBwm.cc \
                       XrdBwmConfig.cc \
                       XrdBwmHandle.cc \
                       XrdBwmLogger.cc \
                       XrdBwmPolicy1.cc

libXrdBwm_la_LIBADD = ../XrdAcc/libXrdAcc.la \
                      ../XrdNet/libXrdNet.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdSys/libXrdSys.la
