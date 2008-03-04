#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.6 2007/08/08 14:23:18 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdPosix.la libXrdPosixPreload.la

# N.B.: XrdPosixXrootd.hh includes XrdSysPthread.hh!
xrootdposixincdir = $(includedir)/xrootd/XrdPosix
xrootdposixinc_HEADERS = XrdPosixXrootd.hh XrdPosixLinkage.hh \
                        XrdPosixExtern.hh  XrdPosixStream.hh

INCLUDES = -I$(top_srcdir)/src

libXrdPosix_la_SOURCES = \
        XrdPosix.cc  \
        XrdPosixLinkage.cc \
        XrdPosixXrootd.cc

libXrdPosix_la_LIBADD = ../XrdClient/libXrdClient.la \
                        ../XrdSec/libXrdSec.la \
                        ../XrdNet/libXrdNet.la \
                        ../XrdOuc/libXrdOuc.la \
                        ../XrdSys/libXrdSys.la

libXrdPosixPreload_la_SOURCES = \
         XrdPosixPreload.cc  \
         XrdPosixLinkage.cc \
         XrdPosixXrootd.cc \
         XrdPosixStream.cc \
         XrdPosix.cc  

libXrdPosixPreload_la_LIBADD = ../XrdClient/libXrdClient.la \
                         ../XrdNet/libXrdNet.la \
                         ../XrdOuc/libXrdOuc.la \
                         ../XrdSec/libXrdSec.la \
                         ../XrdSys/libXrdSys.la
