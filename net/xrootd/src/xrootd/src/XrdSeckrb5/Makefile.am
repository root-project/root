#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.5 2008/09/24 17:01:15 ganis Exp $
## Checked in by $Author: ganis $
#######################################################################

lib_LTLIBRARIES = libXrdSeckrb5.la

INCLUDES = -I$(top_srcdir)/src $(KRB5_INCDIR)

libXrdSeckrb5_la_SOURCES = XrdSecProtocolkrb5.cc
libXrdSeckrb5_la_LIBADD = ../XrdSys/libXrdSys.la ../XrdSys/libXrdOuc.la ../XrdSys/libXrdNet.la \
                          $(KRB5_LIBDIR) -lkrb5 -lk5crypto -lcom_err
libXrdSeckrb5_la_LDFLAGS = -module
libXrdSeckrb5_la_LIBTOOLFLAGS = --tag=disable-static