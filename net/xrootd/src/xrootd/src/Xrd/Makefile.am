#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.7 2008/05/26 13:36:50 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

## Note: XrdObject.hh XrdObject.icc seem not to be used anywhere
EXTRA_DIST = XrdObject.hh XrdObject.icc

xrootdxrdincdir = $(includedir)/xrootd/Xrd

xrootdxrdinc_HEADERS = \
    XrdBuffer.hh  XrdLinkMatch.hh  XrdProtocol.hh \
    XrdConfig.hh  XrdObject.hh     XrdScheduler.hh \
    XrdInet.hh    XrdPoll.hh       XrdStats.hh \
    XrdInfo.hh    XrdPollDev.hh    XrdTrace.hh \
    XrdJob.hh     XrdPollPoll.hh \
    XrdLink.hh    XrdProtLoad.hh \
    XrdObject.icc XrdPollDev.icc XrdPollPoll.icc \
    XrdPollE.hh   XrdPollE.icc

lib_LTLIBRARIES = libXrd.la

INCLUDES = -I$(top_srcdir)/src

libXrd_la_SOURCES = \
        XrdBuffer.cc XrdBuffer.hh     \
        XrdConfig.cc XrdConfig.hh     \
        XrdInet.cc XrdInet.hh       \
        XrdInfo.cc XrdInfo.hh       \
	XrdLinkMatch.cc XrdLinkMatch.hh \
        XrdLink.cc XrdLink.hh       \
        XrdMain.cc        \
        XrdPoll.cc XrdPollDev.hh XrdPollPoll.hh XrdPoll.hh XrdPollPoll.icc XrdPollDev.icc   \
        XrdProtocol.cc XrdProtocol.hh   \
        XrdProtLoad.cc XrdProtLoad.hh \
        XrdScheduler.cc XrdScheduler.hh XrdJob.hh  \
        XrdStats.cc XrdStats.hh \
                    XrdTrace.hh
