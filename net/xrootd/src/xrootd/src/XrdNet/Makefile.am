#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.3 2007/11/16 01:07:38 abh Exp $
## Checked in by $Author: abh $
#######################################################################

lib_LTLIBRARIES = libXrdNet.la

xrootdnetincdir = $(includedir)/xrootd/XrdNet

xrootdnetinc_HEADERS = XrdNet.hh XrdNetOpts.hh XrdNetPeer.hh \
                       XrdNetBuffer.hh XrdNetConnect.hh XrdNetDNS.hh \
		       XrdNetLink.hh XrdNetSecurity.hh XrdNetSocket.hh \
		       XrdNetWork.hh

INCLUDES = -I$(top_srcdir)/src

libXrdNet_la_SOURCES = \
        XrdNet.cc XrdNet.hh XrdNetOpts.hh XrdNetPeer.hh         \
        XrdNetBuffer.cc XrdNetBuffer.hh     \
        XrdNetConnect.cc XrdNetConnect.hh    \
        XrdNetDNS.cc XrdNetDNS.hh        \
        XrdNetLink.cc XrdNetLink.hh       \
        XrdNetMsg.cc XrdNetMsg.hh        \
        XrdNetSecurity.cc XrdNetSecurity.hh   \
        XrdNetSocket.cc XrdNetSocket.hh     \
        XrdNetWork.cc XrdNetWork.hh
libXrdNet_la_LIBADD = $(LIBSOCKET) $(LIBNSL)
#REFAC libXrdNet_la_CXXFLAGS = -prefer-pic
#REFAC libXrdNet_la_LDFLAGS = -static

