#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.8 2010/08/05 15:27:07 feichtinger Exp $
## Checked in by $Author: feichtinger $
#######################################################################

AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra directory definitions for install area
xrootdxrdincdir = $(includedir)/xrootd/Xrd

#############################################################################
## deliverables (to be built / installed)
lib_LTLIBRARIES = libXrd.la

xrootdxrdinc_HEADERS = \
    XrdBuffer.hh  XrdLinkMatch.hh  XrdProtocol.hh \
    XrdConfig.hh  XrdObject.hh     XrdScheduler.hh \
    XrdInet.hh    XrdPoll.hh       XrdStats.hh \
    XrdInfo.hh    XrdPollDev.hh    XrdTrace.hh \
    XrdJob.hh     XrdPollPoll.hh \
    XrdLink.hh    XrdProtLoad.hh \
    XrdObject.icc XrdPollDev.icc XrdPollPoll.icc \
    XrdPollE.hh   XrdPollE.icc


#############################################################################
## Build rules

libXrd_la_SOURCES = \
        XrdBuffer.cc XrdBuffer.hh     \
        XrdConfig.cc XrdConfig.hh     \
        XrdInet.cc XrdInet.hh       \
        XrdInfo.cc XrdInfo.hh       \
        XrdLink.cc XrdLink.hh       \
	XrdLinkMatch.cc XrdLinkMatch.hh \
        XrdMain.cc        \
        XrdPoll.cc XrdPollDev.hh XrdPollPoll.hh XrdPoll.hh \
              XrdPollPoll.icc XrdPollDev.icc   \
        XrdProtLoad.cc XrdProtLoad.hh \
        XrdProtocol.cc XrdProtocol.hh   \
        XrdScheduler.cc XrdScheduler.hh XrdJob.hh  \
        XrdStats.cc XrdStats.hh \
                    XrdTrace.hh


## Extra sources that should go into the tarball
##       Note: XrdObject.hh XrdObject.icc seem not to be used anywhere
EXTRA_DIST = XrdObject.hh XrdObject.icc
