#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 17.3.2009
##
## Version info: $Id$
## Checked in by $Author: feichtinger $
#######################################################################

#Note: xrdadler32 unfortunately has a dependency on the XrdPosix library


#############################################################################
## extra directory definitions


#############################################################################
## deliverables

bin_PROGRAMS = xrdadler32 mpxstats wait41


#############################################################################
## Build rules

AM_CPPFLAGS = -I$(top_srcdir)/src

xrdadler32_SOURCES = Xrdadler32.cc

xrdadler32_LDADD = \
                       ../XrdClient/libXrdClient.la  \
                       ../XrdPosix/libXrdPosix.la   \
                       ../XrdOuc/libXrdOuc.la     \
                       ../XrdSys/libXrdSys.la

xrdadler32_LDFLAGS = -static -export-dynamic -ldl -lz

mpxstats_SOURCES = XrdMpxStats.cc

mpxstats_LDADD = \
                       ../XrdNet/libXrdNet.la   \
                       ../XrdOuc/libXrdOuc.la     \
                       ../XrdSys/libXrdSys.la \
                       ../XrdNet/libXrdNetUtil.la

mpxstats_LDFLAGS = -static -export-dynamic

wait41_SOURCES = XrdWait41.cc

wait41_LDADD = \
                       ../XrdOuc/libXrdOuc.la     \
                       ../XrdSys/libXrdSys.la

wait41_LDFLAGS = -static -export-dynamic
