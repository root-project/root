#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 17.5.2008
##
## Version info: $Id$
## Checked in by $Author: dfeich $
#######################################################################

#############################################################################
## extra directory definitions
xrootdfrmincdir = $(includedir)/xrootd/XrdFrm

#############################################################################
## deliverables

bin_PROGRAMS = frm_pstga frm_pstgd

xrootdfrminc_HEADERS = XrdFrmConfig.hh  \
                       XrdFrmPstg.hh    \
                       XrdFrmPstgReq.hh \
                       XrdFrmPstgXfr.hh

# convenience library to help in the build
noinst_LTLIBRARIES = libXrdFrm.la


#############################################################################
## Build rules

AM_CPPFLAGS = -I$(top_srcdir)/src

libXrdFrm_la_SOURCES = \
                  XrdFrmConfig.cc  \
                  XrdFrmPstg.cc    \
                  XrdFrmPstgMain.cc\
                  XrdFrmPstgReq.cc \
                  XrdFrmPstgXfr.cc

libXrdFrm_la_LIBADD = \
                  ../XrdOss/libXrdOss.la     \
                  ../XrdNet/libXrdNet.la     \
                  ../XrdOuc/libXrdOuc.la     \
                  ../XrdSys/libXrdSys.la

# Note: The build rules for pgsta and pstgd are identical. The programs
# behave differently based on their invocation name

frm_pstga_SOURCES = \
                    ../XrdXrootd/XrdXrootdMonitor.cc \
                    ../Xrd/XrdScheduler.cc
frm_pstga_LDADD = libXrdFrm.la
frm_pstga_LDFLAGS = -static -export-dynamic -ldl


frm_pstgd_SOURCES = \
                    ../XrdXrootd/XrdXrootdMonitor.cc \
                    ../Xrd/XrdScheduler.cc
frm_pstgd_LDADD = libXrdFrm.la
frm_pstgd_LDFLAGS = -static -export-dynamic -ldl


