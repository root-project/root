#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 17.5.2008
##
## Version info: $Id$
## Checked in by $Author: feichtinger $
#######################################################################

#
# WARNING: In this Makefile there are a number of nonideal solutions
#          that needed to be implemented. It must get reorganized
#          For the moment, please ask derek.feichtinger@psi.ch
#          before making nontrivial changes
#


AM_CPPFLAGS = -I$(top_srcdir)/src $(READLINE_INCDIR)

#############################################################################
## extra directory definitions
xrootdfrmincdir = $(includedir)/xrootd/XrdFrm

#############################################################################
## deliverables

bin_PROGRAMS = frm_admin frm_purged frm_xfragent frm_xfrd

xrootdfrminc_HEADERS = XrdFrmAdmin.hh   \
                       XrdFrmCID.hh     \
                       XrdFrmConfig.hh  \
                       XrdFrmFiles.hh   \
                       XrdFrmMigrate.hh \
                       XrdFrmMonitor.hh \
                       XrdFrmProxy.hh   \
                       XrdFrmPurge.hh   \
                       XrdFrmReqAgent.hh \
                       XrdFrmReqBoss.hh \
                       XrdFrmReqFile.hh \
                       XrdFrmRequest.hh \
                       XrdFrmTrace.hh   \
                       XrdFrmTSort.hh   \
                       XrdFrmTransfer.hh\
                       XrdFrmUtils.hh   \
                       XrdFrmXfrAgent.hh\
                       XrdFrmXfrDaemon.hh\
                       XrdFrmXfrJob.hh  \
                       XrdFrmXfrQueue.hh


lib_LTLIBRARIES = libXrdFrm.la

# Convenience libs to use for objs common to multiple binaries in this dir
noinst_LTLIBRARIES = libXrdFrmConv.la libXrdFrmConvAgent.la


#############################################################################
## Build rules


libXrdFrmConv_la_SOURCES = XrdFrmConfig.cc

libXrdFrmConvAgent_la_SOURCES = XrdFrmReqBoss.cc   \
              XrdFrmTransfer.cc \
              XrdFrmXfrAgent.cc \
              XrdFrmXfrDaemon.cc \
              XrdFrmMigrate.cc  \
              XrdFrmXfrQueue.cc


libXrdFrm_la_SOURCES = \
                  XrdFrmCID.cc     \
                  XrdFrmFiles.cc  \
                  XrdFrmMonitor.cc\
                  XrdFrmProxy.cc  \
                  XrdFrmReqAgent.cc \
                  XrdFrmReqFile.cc \
                  XrdFrmTrace.cc   \
                  XrdFrmTSort.cc \
                  XrdFrmUtils.cc

libXrdFrm_la_LIBADD = \
                  ../XrdNet/libXrdNet.la     \
                  ../XrdOuc/libXrdOuc.la     \
                  ../XrdSys/libXrdSys.la     \
                  ../XrdNet/libXrdNetUtil.la \
                  ../XrdOss/libXrdOss.la \
                  ../Xrd/libXrd.la $(READLINE_LIBDIR) $(READLINE_LIBS)

libXrdFrm_la_LDFLAGS = -static


# $(ECHO)$(LD) $(LDOP) $(OBJADMIN) $(OBJADDS) $(LIBREADLINE) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINADMIN)
frm_admin_SOURCES = \
                  XrdFrmAdminMain.cc \
                  XrdFrmAdmin.cc \
                  XrdFrmAdminAudit.cc \
                  XrdFrmAdminFiles.cc \
                  XrdFrmAdminFind.cc \
                  XrdFrmAdminQuery.cc \
                  XrdFrmAdminUnlink.cc


frm_admin_LDADD = ../XrdOss/libXrdOss.la \
                  libXrdFrmConv.la \
                  libXrdFrm.la

frm_admin_LDFLAGS = -static -export-dynamic -ldl


# $(ECHO)$(LD) $(LDOP) $(OBJPURG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINPURGD)
frm_purged_SOURCES = \
                    XrdFrmPurgMain.cc \
                    XrdFrmPurge.cc

frm_purged_LDADD = libXrdFrmConv.la \
                   libXrdFrm.la \
                   ../XrdOuc/libXrdOuc.la   XrdFrmProxy.o

frm_purged_LDFLAGS = -static -export-dynamic -ldl

# Note: The build rules for xfragent and xfrd are identical. The programs
# behave differently based on their invocation name

#$(ECHO)$(LD) $(LDOP) $(OBJXFR) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINXFRD)
frm_xfragent_SOURCES = XrdFrmXfrMain.cc

frm_xfragent_LDADD = libXrdFrmConv.la \
                     libXrdFrmConvAgent.la \
                     libXrdFrm.la \
                   ../XrdOuc/libXrdOuc.la   XrdFrmProxy.o

frm_xfragent_LDFLAGS = -static -export-dynamic -ldl


frm_xfrd_SOURCES = XrdFrmXfrMain.cc

frm_xfrd_LDADD =     libXrdFrmConv.la \
                     libXrdFrmConvAgent.la \
                     libXrdFrm.la \
                   ../XrdOuc/libXrdOuc.la   XrdFrmProxy.o

frm_xfrd_LDFLAGS = -static -export-dynamic -ldl

