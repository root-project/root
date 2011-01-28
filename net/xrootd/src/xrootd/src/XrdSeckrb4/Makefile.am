#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.3 2007/06/07 10:44:59 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdSeckrb4.la

INCLUDES = -I$(top_srcdir)/src $(KRB4_INCDIR)

libXrdSeckrb4_la_SOURCES = XrdSecProtocolkrb4.cc
libXrdSeckrb4_la_LIBADD = $(KRB5_LIBDIR) -lkrb4 -lkrb5 -ldes425 -lk5crypto -lcom_err
libXrdSeckrb4_la_LDFLAGS = -module
libXrdSeckrb4_la_LIBTOOLFLAGS = --tag=disable-static