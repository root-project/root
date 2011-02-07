#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

lib_LTLIBRARIES = libXrdSecunix.la

INCLUDES = -I$(top_srcdir)/src $(KRB4_INCDIR)

libXrdSecunix_la_SOURCES = XrdSecProtocolunix.cc
libXrdSecunix_la_LIBADD = ../XrdSec/libXrdSec.la \
                          ../XrdOuc/libXrdOuc.la \
                          ../XrdSys/libXrdSys.la

libXrdSecunix_la_LDFLAGS = -module
libXrdSecunix_la_LIBTOOLFLAGS = --tag=disable-static
