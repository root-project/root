#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra installation directory definitions

xrootdoucincdir = $(includedir)/xrootd/XrdOuc

#############################################################################
## deliverables (to be built / installed)

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


#############################################################################
# Build rules

libXrdOuc_la_SOURCES = \
        XrdOuca2x.cc XrdOuca2x.hh        \
        XrdOucArgs.cc XrdOucArgs.hh        \
        XrdOucCRC.cc XrdOucCRC.hh \
        XrdOucEnv.cc XrdOucEnv.hh XrdOucHash.hh       \
	XrdOucExport.cc XrdOucExport.hh \
        XrdOucHashVal.cc     \
        XrdOucMsubs.cc XrdOucMsubs.hh \
        XrdOucName2Name.cc XrdOucName2Name.hh \
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
        XrdOucUtils.cc XrdOucUtils.hh   

#        XrdOucCRC.cc XrdOucCRC.hh \
#        XrdOucHash.icc


libXrdOuc_la_LIBADD = $(LIBPTHREAD) $(SEM_LIBRT) $(LIBNSL) ../XrdSys/libXrdSysErrLog.la

# sources not named in rules that still need to get rolled into the distribution tarball
EXTRA_DIST = XrdOucRash.hh XrdOucRash.icc


# TODO: (DEREK) why are these compiled in the classic GNUmakefile instead of making a lib?
#        XrdSys/XrdSysError.cc XrdSys/XrdSysError.hh \
#        XrdSys/XrdSysLogger.cc 
# TODO: again as in classic? REFAC libXrdOuc_la_CXXFLAGS = -prefer-pic
# TODO: again as in classic? REFAC libXrdOuc_la_LDFLAGS = -static
