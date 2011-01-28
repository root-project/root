#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.2 2007/06/07 10:45:05 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

noinst_LTLIBRARIES = libXrdSfs.la

xrootdsfsincdir = $(includedir)/xrootd/XrdSfs

xrootdsfsinc_HEADERS = XrdSfsAio.hh  XrdSfsInterface.hh  XrdSfsNative.hh

INCLUDES = -I$(top_srcdir)/src

libXrdSfs_la_SOURCES = \
         XrdSfsNative.cc XrdSfsNative.hh XrdSfsAio.hh XrdSfsInterface.hh
