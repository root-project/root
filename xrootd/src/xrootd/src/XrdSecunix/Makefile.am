#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.1 2007/08/02 01:41:05 abh Exp $
## Checked in by $Author: abh $
#######################################################################

lib_LTLIBRARIES = libXrdSecunix.la

INCLUDES = -I$(top_srcdir)/src $(KRB4_INCDIR)

libXrdSecunix_la_SOURCES = XrdSecProtocolunix.cc
libXrdSecunix_la_LIBADD =
libXrdSecunix_la_LDFLAGS = -module
libXrdSecunix_la_LIBTOOLFLAGS = --tag=disable-static
