#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

noinst_LTLIBRARIES = libXrdSfs.la

xrootdsfsincdir = $(includedir)/xrootd/XrdSfs

xrootdsfsinc_HEADERS = XrdSfsAio.hh  XrdSfsInterface.hh  XrdSfsNative.hh

INCLUDES = -I$(top_srcdir)/src -I$(top_builddir)/src

libXrdSfs_la_SOURCES = \
         XrdSfsNative.cc XrdSfsNative.hh XrdSfsAio.hh XrdSfsInterface.hh
