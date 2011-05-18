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

xrootdnetincdir = $(includedir)/xrootd/XrdNet

#############################################################################
## deliverables (to be built / installed)

lib_LTLIBRARIES = libXrdNet.la libXrdNetUtil.la 


xrootdnetinc_HEADERS = XrdNet.hh XrdNetOpts.hh XrdNetPeer.hh \
                       XrdNetBuffer.hh XrdNetConnect.hh XrdNetDNS.hh \
		       XrdNetLink.hh XrdNetSecurity.hh XrdNetSocket.hh \
		       XrdNetWork.hh XrdNetCmsNotify.hh

#############################################################################
# Build rules

libXrdNet_la_SOURCES = \
        XrdNet.cc XrdNet.hh XrdNetOpts.hh XrdNetPeer.hh         \
        XrdNetBuffer.cc XrdNetBuffer.hh     \
        XrdNetCmsNotify.cc XrdNetCmsNotify.hh \
        XrdNetConnect.cc XrdNetConnect.hh    \
        XrdNetLink.cc XrdNetLink.hh       \
        XrdNetMsg.cc XrdNetMsg.hh        \
        XrdNetSecurity.cc XrdNetSecurity.hh   \
        XrdNetSocket.cc XrdNetSocket.hh     \
        XrdNetWork.cc XrdNetWork.hh

libXrdNet_la_LIBADD = $(LIBSOCKET) $(LIBNSL)


libXrdNetUtil_la_SOURCES = \
        XrdNetDNS.cc XrdNetDNS.hh

libXrdNetUtil_la_LIBADD = $(LIBSOCKET) $(LIBNSL)


#REFAC libXrdNet_la_CXXFLAGS = -prefer-pic
#REFAC libXrdNet_la_LDFLAGS = -static

