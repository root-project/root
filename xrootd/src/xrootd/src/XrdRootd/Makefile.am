#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.1 2005/10/06 15:30:22 ganis Exp $
## Checked in by $Author: ganis $
#######################################################################

lib_LTLIBRARIES = libXrdRootd.la

INCLUDES = -I$(top_srcdir)/src

libXrdRootd_la_SOURCES = XrdRootdProtocol.cc XrdRootdProtocol.hh
libXrdRootd_la_LIBADD = ../XrdOuc/libXrdOuc.la
