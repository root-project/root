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
xrootdfrmincdir = $(includedir)/xrootd/XrdFrm

#############################################################################
## deliverables

bin_PROGRAMS = frm_admin frm_purged frm_xfragent frm_xfrd

xrootdfrminc_HEADERS = XrdFrmAdmin.hh   \
                       XrdFrmConfig.hh  \
                       XrdFrmFiles.hh   \
                       XrdFrmMigrate.hh \
                       XrdFrmMonitor.hh \
                       XrdFrmProxy.hh   \
                       XrdFrmPurge.hh   \
                       XrdFrmReqBoss.hh \
                       XrdFrmReqFile.hh \
                       XrdFrmRequest.hh \
                       XrdFrmTrace.hh   \
                       XrdFrmTsort.hh   \
                       XrdFrmTransfer.hh\
                       XrdFrmUtils.hh   \
                       XrdFrmXfrAgent.hh\
                       XrdFrmXfrDaemon.hh\
                       XrdFrmXfrJob.hh  \
                       XrdFrmXfrQueue.hh

# convenience library to help in the build
noinst_LTLIBRARIES = libXrdFrm.la


#############################################################################
## Build rules

AM_CPPFLAGS = -I$(top_srcdir)/src $(READLINE_INCDIR)

libXrdFrm_la_SOURCES = \
                  XrdFrmAdmin.cc       \
                  XrdFrmAdminAudit.cc  \
                  XrdFrmAdminFiles.cc  \
                  XrdFrmAdminFind.cc   \
                  XrdFrmAdminQuery.cc  \
                  XrdFrmAdminReloc.cc  \
                  XrdFrmAdminUnlink.cc \
                  XrdFrmConfig.cc  \
                  XrdFrmFiles.cc   \
                  XrdFrmMonitor.cc\
                  XrdFrmMigrate.cc\
                  XrdFrmProxy.cc  \
                  XrdFrmPurge.cc  \
                  XrdFrmReqBoss.cc \
                  XrdFrmReqFile.cc \
                  XrdFrmTrace.cc   \
                  XrdFrmTransfer.cc\
                  XrdFrmTSort.cc   \
                  XrdFrmUtils.cc   \
                  XrdFrmXfrAgent.cc
                  XrdFrmXfrDaemon.cc
                  XrdFrmXfrQueue.cc

libXrdFrm_la_LIBADD = \
                  ../XrdOss/libXrdOss.la     \
                  ../XrdNet/libXrdNet.la     \
                  ../XrdOuc/libXrdOuc.la     \
                  ../XrdSys/libXrdSys.la     \
                  ../Xrd/libXrd.la $(READLINE_LIBDIR) $(READLINE_LIBS)

frm_admin_SOURCES = XrdFrmAdminMain.cc
frm_admin_LDADD = libXrdFrm.la
frm_admin_LDFLAGS = -static -export-dynamic -ldl

frm_purged_SOURCES = \
                    XrdFrmPurgMain.cc \
                    XrdFrmPurge.cc
frm_purged_LDADD = libXrdFrm.la
frm_purged_LDFLAGS = -static -export-dynamic -ldl

# Note: The build rules for xfragent and xfrd are identical. The programs
# behave differently based on their invocation name

frm_xfragent_SOURCES = \
                    XrdFrmXfrMain.cc
frm_xfragent_LDADD = libXrdFrm.la
frm_xfragent_LDFLAGS = -static -export-dynamic -ldl

frm_xfrd_SOURCES = \
                    XrdFrmXfrMain.cc
frm_xfrd_LDADD = libXrdFrm.la
frm_xfrd_LDFLAGS = -static -export-dynamic -ldl
