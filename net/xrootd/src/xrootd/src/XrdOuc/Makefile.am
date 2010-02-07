#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.15 2009/08/19 13:51:46 abh Exp $
## Checked in by $Author: abh $
#######################################################################

EXTRA_DIST = XrdOucRash.hh XrdOucRash.icc

xrootdoucincdir = $(includedir)/xrootd/XrdOuc

xrootdoucinc_HEADERS = \
    XrdOuca2x.hh    XrdOucChain.hh      XrdOucDLlist.hh \
    XrdOucEnv.hh    XrdOucErrInfo.hh    XrdOucExport.hh \
    XrdOucHash.hh   XrdOucHash.icc      XrdOucLock.hh  \
    XrdOucMsubs.hh  XrdOucName2Name.hh  XrdOucNList.hh  \
    XrdOucPList.hh  XrdOucRash.hh       XrdOucRash.icc \
    XrdOucReqID.hh  XrdOucStream.hh     XrdOucString.hh \
    XrdOucTable.hh  XrdOucTList.hh      XrdOucTokenizer.hh \
    XrdOucTrace.hh  XrdOucUtils.hh      XrdOucProg.hh \
    XrdOucPup.hh    XrdOucCRC.hh        XrdOucArgs.hh \
    XrdOucNSWalk.hh XrdOucSxeq.hh

lib_LTLIBRARIES = libXrdOuc.la

INCLUDES = -I$(top_srcdir)/src

libXrdOuc_la_SOURCES = \
        XrdOuca2x.cc XrdOuca2x.hh        \
        XrdOucArgs.cc XrdOucArgs.hh        \
        XrdOucEnv.cc XrdOucEnv.hh XrdOucHash.hh       \
        XrdOucName2Name.cc XrdOucName2Name.hh \
        XrdOucHashVal.cc     \
        XrdOucMsubs.cc XrdOucMsubs.hh \
        XrdOucNList.cc XrdOucNList.hh      \
        XrdOucNSWalk.cc XrdOucNSWalk.hh      \
        XrdOucProg.cc XrdOucProg.hh \
        XrdOucPup.cc XrdOucPup.hh \
        XrdOucReqID.cc XrdOucReqID.hh      \
        XrdOucStream.cc XrdOucStream.hh     \
        XrdOucString.cc XrdOucString.hh     \
        XrdOucSxeq.cc XrdOucSxeq.hh \
        XrdOucTokenizer.cc XrdOucTokenizer.hh  \
        XrdOucTrace.cc XrdOucTrace.hh      \
        XrdOucUtils.cc XrdOucUtils.hh      \
	XrdOucExport.cc XrdOucExport.hh \
        XrdOucCRC.cc XrdOucCRC.hh \
        XrdOucHash.icc

libXrdOuc_la_LIBADD = $(LIBPTHREAD) $(SEM_LIBRT) $(LIBNSL)
#REFAC libXrdOuc_la_CXXFLAGS = -prefer-pic
#REFAC libXrdOuc_la_LDFLAGS = -static

