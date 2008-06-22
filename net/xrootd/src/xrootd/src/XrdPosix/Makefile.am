#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.10 2008/03/18 23:16:49 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

lib_LTLIBRARIES = libXrdPosix.la libXrdPosixPreload.la

# N.B.: XrdPosixXrootd.hh includes XrdSysPthread.hh!
xrootdposixincdir = $(includedir)/xrootd/XrdPosix
xrootdposixinc_HEADERS = XrdPosixXrootd.hh XrdPosixOsDep.hh XrdPosixLinkage.hh \
                        XrdPosix.hh XrdPosixExtern.hh


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
         XrdPosixPreload32.cc  \
         XrdPosixLinkage.cc \
         XrdPosixXrootd.cc \
         XrdPosix.cc

# TODO: What to do with XrdPosixPreload32.cc?

libXrdPosixPreload_la_LIBADD = ../XrdClient/libXrdClient.la \
                         ../XrdNet/libXrdNet.la \
                         ../XrdOuc/libXrdOuc.la \
                         ../XrdSec/libXrdSec.la \
                         ../XrdSys/libXrdSys.la

