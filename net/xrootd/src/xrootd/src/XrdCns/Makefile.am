#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 17.5.2008
##
## Version info: $Id$
## Checked in by $Author: abh $
#######################################################################

#############################################################################
## extra directory definitions
xrootdcnsincdir = $(includedir)/xrootd/XrdCns

#############################################################################
## deliverables
bin_PROGRAMS = XrdCnsd

INCLUDES = -I$(top_srcdir)/src

xrootdcnsinc_HEADERS = XrdCnsDaemon.hh \
                       XrdCnsEvent.hh

#############################################################################
## Build rules

XrdCnsd_SOURCES = \
                  XrdCnsConfig.cc  \
                  XrdCnsDaemon.cc  \
                  XrdCnsEvent.cc   \
                  XrdCnsMain.cc

XrdCnsd_LDADD   = \
                  libXrd.la        \
                  libXrdPosix.la   \
                  libXrdClient.la  \
                  libXrdNet.la     \
                  libXrdOuc.la     \
                  libXrdSys.la

XrdCnsd_LDFLAGS = -static -export-dynamic

$(OBJECT_XRDCNSD) $(BINLIBS) $(LIBM) $(LIBS)
