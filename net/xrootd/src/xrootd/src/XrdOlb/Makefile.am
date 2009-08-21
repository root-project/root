#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.13 2009/06/24 21:24:03 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

bin_PROGRAMS = olbd

xrootdolbincdir = $(includedir)/xrootd/XrdOlb

xrootdolbinc_HEADERS = \
    XrdOlbAdmin.hh    XrdOlbPList.hh     XrdOlbServer.hh \
    XrdOlbCache.hh    XrdOlbPrepArgs.hh  XrdOlbState.hh \
    XrdOlbConfig.hh   XrdOlbPrepare.hh   XrdOlbTrace.hh \
    XrdOlbManList.hh  XrdOlbProtocol.hh  XrdOlbTypes.hh \
    XrdOlbManTree.hh  XrdOlbRRQ.hh       XrdOlbXmi.hh \
    XrdOlbManager.hh  XrdOlbRTable.hh    XrdOlbXmiReq.hh \
    XrdOlbMeter.hh    XrdOlbReq.hh


INCLUDES = -I$(top_srcdir)/src

olbd_SOURCES = \
        XrdOlbAdmin.cc XrdOlbAdmin.hh XrdOlbTrace.hh   \
        XrdOlbCache.cc XrdOlbCache.hh    \
        XrdOlbConfig.cc XrdOlbConfig.hh XrdOlbTypes.hh  \
        XrdOlbManager.cc XrdOlbManager.hh  \
        XrdOlbManList.cc XrdOlbManList.hh  \
	XrdOlbManTree.cc XrdOlbManTree.hh \
	XrdOlbPrepArgs.cc XrdOlbPrepArgs.hh \
        XrdOlbMeter.cc XrdOlbMeter.hh    \
        XrdOlbPList.cc XrdOlbPList.hh    \
        XrdOlbPrepare.cc XrdOlbPrepare.hh  \
        XrdOlbProtocol.cc XrdOlbProtocol.hh \
	XrdOlbReq.cc XrdOlbReq.hh \
        XrdOlbRRQ.cc XrdOlbRRQ.hh \
        XrdOlbRTable.cc XrdOlbRTable.hh \
        XrdOlbServer.cc XrdOlbServer.hh   \
        XrdOlbState.cc XrdOlbState.hh \
	XrdOlbXmiReq.cc XrdOlbXmiReq.hh XrdOlbXmi.hh

olbd_LDADD = ../Xrd/libXrd.la \
             ../XrdNet/libXrdNet.la \
             ../XrdOuc/libXrdOuc.la \
             ../XrdSys/libXrdSys.la \
             -ldl

olbd_LDFLAGS = -static -export-dynamic


