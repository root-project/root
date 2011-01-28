#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 10.6.2008
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

# preprocessor flags
AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra directory definitions for install area
Secsssincdir = $(includedir)/xrootd/XrdSecsss

#############################################################################
## deliverables (to be built / installed)
lib_LTLIBRARIES = libXrdSecsss.la

bin_PROGRAMS = xrdsssadmin

Secsssinc_HEADERS = XrdSecProtocolsss.hh  XrdSecsssID.hh  XrdSecsssKT.hh  XrdSecsssRR.hh

#############################################################################
## Build rules
libXrdSecsss_la_SOURCES = \
   XrdSecProtocolsss.cc \
   XrdSecsssID.cc \
   XrdSecsssKT.cc

libXrdSecsss_la_LIBADD = ../XrdCrypto/libXrdCryptoLite.la \
                      ../XrdOuc/libXrdOuc.la \
                      ../XrdNet/libXrdNet.la \
                      ../XrdSys/libXrdSys.la \
                      ../XrdNet/libXrdNetUtil.la

libXrdSecsss_la_LDFLAGS = -module $(CRYPTLIB) $(OPENSSL_LIBDIR)

xrdsssadmin_SOURCES = XrdSecsssAdmin.cc
xrdsssadmin_LDFLAGS = -static
xrdsssadmin_LDADD =   libXrdSecsss.la

#                      ../XrdCrypto/libXrdCryptoLite.la \
#                      ../XrdOuc/libXrdOuc.la \
#                      ../XrdNet/libXrdNet.la \
#                      ../XrdSys/libXrdSys.la \
#                     ../XrdNet/libXrdNetUtil.la \
#		      $(CRYPTLIB) $(OPENSSL_LIBDIR) -lcrypto


