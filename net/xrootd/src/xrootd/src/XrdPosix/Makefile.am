#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

include $(top_srcdir)/src/Makefile_include

# preprocessor flags
AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/src

#############################################################################
## extra directory definitions for install area
# N.B.: XrdPosixXrootd.hh includes XrdSysPthread.hh!
xrootdposixincdir = $(includedir)/xrootd/XrdPosix
xrootdposixinc_HEADERS = XrdPosixXrootd.hh XrdPosixOsDep.hh XrdPosixLinkage.hh \
                        XrdPosix.hh XrdPosixExtern.hh XrdPosixCallBack.hh

#############################################################################
## deliverables

lib_LTLIBRARIES = libXrdPosix.la libXrdPosixPreload.la

#############################################################################
## Build rules

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
         XrdPosixPreload32.cc \
         XrdPosixPreload.cc  \
         XrdPosixLinkage.cc \
         XrdPosixXrootd.cc \
         XrdPosix.cc

libXrdPosixPreload_la_LIBADD = \
                         ../XrdClient/libXrdClient.la \
                         ../XrdNet/libXrdNet.la \
                         ../XrdOuc/libXrdOuc.la \
                         ../XrdSec/libXrdSec.la \
                         ../XrdSys/libXrdSys.la

