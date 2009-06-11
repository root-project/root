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

bin_PROGRAMS = frm_admin frm_pstga frm_pstgd

xrootdfrminc_HEADERS = XrdFrmAdmin.hh   \
                       XrdFrmConfig.hh  \
                       XrdFrmFiles.hh   \
                       XrdFrmPstg.hh    \
                       XrdFrmPstgReq.hh \
                       XrdFrmPstgXfr.hh \
                       XrdFrmTrace.hh   \
                       XrdFrmUtils.hh

# convenience library to help in the build
noinst_LTLIBRARIES = libXrdFrm.la


#############################################################################
## Build rules

AM_CPPFLAGS = -I$(top_srcdir)/src $(READLINE_INCDIR)

libXrdFrm_la_SOURCES = \
                  XrdFrmAdminAudit.cc  \
                  XrdFrmAdminFiles.cc  \
                  XrdFrmAdminFind.cc   \
                  XrdFrmAdminMain.cc   \
                  XrdFrmAdminQuery.cc  \
                  XrdFrmAdminReloc.cc  \
                  XrdFrmAdminUnlink.cc \
                  XrdFrmConfig.cc  \
                  XrdFrmFiles.cc   \
                  XrdFrmPstg.cc    \
                  XrdFrmPstgMain.cc\
                  XrdFrmPstgReq.cc \
                  XrdFrmPstgXfr.cc \
                  XrdFrmUtils.cc

libXrdFrm_la_LIBADD = \
                  ../XrdOss/libXrdOss.la     \
                  ../XrdNet/libXrdNet.la     \
                  ../XrdOuc/libXrdOuc.la     \
                  ../XrdSys/libXrdSys.la     \
                  ../XrdCms/libXrdCms.la     \
                  ../Xrd/libXrd.la $(READLINE_LIBDIR) $(READLINE_LIBS)

frm_admin_SOURCES = XrdFrmAdmin.cc \
                    ../XrdXrootd/XrdXrootdMonitor.cc
frm_admin_LDADD = libXrdFrm.la
frm_admin_LDFLAGS = -static -export-dynamic -ldl

# Note: The build rules for pgsta and pstgd are identical. The programs
# behave differently based on their invocation name

frm_pstga_SOURCES = \
                    XrdFrmAdmin.cc \
                    ../XrdXrootd/XrdXrootdMonitor.cc \
                    ../Xrd/XrdScheduler.cc
frm_pstga_LDADD = libXrdFrm.la
frm_pstga_LDFLAGS = -static -export-dynamic -ldl


frm_pstgd_SOURCES = \
                    XrdFrmAdmin.cc \
                    ../XrdXrootd/XrdXrootdMonitor.cc \
                    ../Xrd/XrdScheduler.cc
frm_pstgd_LDADD = libXrdFrm.la
frm_pstgd_LDFLAGS = -static -export-dynamic -ldl


