#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

#####################################################
# TARGETS
bin_PROGRAMS = cmsd

lib_LTLIBRARIES = libXrdCms.la

# convenience library for the common sources, just used for the build
noinst_LTLIBRARIES = libXrdCmsConv.la 

#####################################################
# HEADERS to be installed
xrootdcmsincdir = $(includedir)/xrootd/XrdCms

xrootdcmsinc_HEADERS = \
XrdCmsAdmin.hh         XrdCmsManTree.hh   XrdCmsReq.hh \
XrdCmsCache.hh         XrdCmsManager.hh   XrdCmsResp.hh \
XrdCmsClient.hh        XrdCmsMeter.hh     XrdCmsRouting.hh \
XrdCmsClientConfig.hh  XrdCmsNash.hh      XrdCmsNode.hh \
XrdCmsKey.hh           XrdCmsSecurity.hh \
XrdCmsClientMan.hh     XrdCmsPList.hh     XrdCmsSelect.hh \
XrdCmsClientMsg.hh     XrdCmsParser.hh    XrdCmsState.hh \
XrdCmsCluster.hh       XrdCmsPrepArgs.hh  XrdCmsSupervisor.hh \
XrdCmsConfig.hh        XrdCmsPrepare.hh   XrdCmsTalk.hh \
XrdCmsFinder.hh        XrdCmsProtocol.hh  XrdCmsTrace.hh \
XrdCmsJob.hh           XrdCmsRRData.hh    XrdCmsTypes.hh \
XrdCmsLogin.hh         XrdCmsRRQ.hh       XrdCmsXmi.hh \
XrdCmsManList.hh       XrdCmsRTable.hh    XrdCmsXmiReq.hh



######################################################
# BUILD DEFINITIONS (SOURCES AND FLAGS)

AM_CPPFLAGS = -I$(top_srcdir)/src

# for convenience:
libXrdCmsConv_la_SOURCES = \
        XrdCmsLogin.cc       \
        XrdCmsParser.cc      \
        XrdCmsRRData.cc      \
        XrdCmsSecurity.cc    \
        XrdCmsTalk.cc


libXrdCms_la_SOURCES = \
        XrdCmsClientConfig.cc\
        XrdCmsClientMan.cc   \
        XrdCmsClientMsg.cc   \
        XrdCmsFinder.cc      \
        XrdCmsResp.cc


libXrdCms_la_LIBADD = libXrdCmsConv.la \
	             ../XrdNet/libXrdNet.la \
		     ../XrdOuc/libXrdOuc.la \
		     ../XrdSys/libXrdSys.la 

cmsd_SOURCES = \
        XrdCmsAdmin.cc       \
        XrdCmsCache.cc       \
        XrdCmsCluster.cc     \
        XrdCmsConfig.cc      \
        XrdCmsJob.cc         \
        XrdCmsKey.cc         \
        XrdCmsManList.cc     \
        XrdCmsManTree.cc     \
        XrdCmsManager.cc     \
        XrdCmsMeter.cc       \
        XrdCmsNash.cc        \
        XrdCmsNode.cc        \
        XrdCmsPList.cc       \
        XrdCmsPrepArgs.cc    \
        XrdCmsPrepare.cc     \
        XrdCmsProtocol.cc    \
        XrdCmsReq.cc         \
        XrdCmsRouting.cc     \
        XrdCmsRRQ.cc         \
        XrdCmsRTable.cc      \
        XrdCmsState.cc       \
        XrdCmsSupervisor.cc  \
        XrdCmsXmiReq.cc



cmsd_LDADD = libXrdCmsConv.la \
             ../XrdNet/libXrdNet.la \
             ../XrdOss/libXrdOss.la \
             ../XrdFrm/libXrdFrm.la \
             ../XrdOuc/libXrdOuc.la \
             ../XrdSys/libXrdSys.la \
             ../Xrd/libXrd.la \
             -ldl

cmsd_LDFLAGS = -static -export-dynamic


