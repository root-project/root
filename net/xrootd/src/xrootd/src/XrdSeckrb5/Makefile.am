#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.9 2008/11/19 17:10:59 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdSeckrb5.la

INCLUDES = -I$(top_srcdir)/src $(KRB5_INCDIR)

libXrdSeckrb5_la_SOURCES = XrdSecProtocolkrb5.cc
libXrdSeckrb5_la_LIBADD = \
              ../XrdSys/libXrdSys.la \
              ../XrdOuc/libXrdOuc.la \
              ../XrdNet/libXrdNet.la \
              $(KRB5_LIBDIR) -lkrb5 -lk5crypto -lcom_err

libXrdSeckrb5_la_LDFLAGS = -module
libXrdSeckrb5_la_LIBTOOLFLAGS = --tag=disable-static
