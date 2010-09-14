#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.14 2010/08/06 15:12:23 feichtinger Exp $
## Checked in by $Author: feichtinger $
#######################################################################

# TODO: solve libXrdPosixPreload build problem. The classic build "solves" the
#       multiply defined symbol problems between XrdPosixPreload.cc and
#       XrdPosixPreload32.cc by compiling the latter with -fPIC, so compiler
#       behavior somehow deals with the symbol clash. This is surely a source
#       of portability problems. I do not yet know hot to best solve it here

include $(top_srcdir)/src/Makefile_include

# preprocessor flags
AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra directory definitions for install area
# N.B.: XrdPosixXrootd.hh includes XrdSysPthread.hh!
xrootdposixincdir = $(includedir)/xrootd/XrdPosix
xrootdposixinc_HEADERS = XrdPosixXrootd.hh XrdPosixOsDep.hh XrdPosixLinkage.hh \
                        XrdPosix.hh XrdPosixExtern.hh XrdPosixCallBack.hh

#############################################################################
## deliverables

lib_LTLIBRARIES = libXrdPosix.la
#lib_LTLIBRARIES = libXrdPosix.la libXrdPosixPreload.la

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

#libXrdPosixPreload_la_SOURCES = \
#         XrdPosixPreload32.cc \
#         XrdPosixPreload.cc  \
#         XrdPosixLinkage.cc \
#         XrdPosixXrootd.cc \
#         XrdPosix.cc

#libXrdPosixPreload_la_LIBADD = \
#                         ../XrdClient/libXrdClient.la \
#                         ../XrdNet/libXrdNet.la \
#                         ../XrdOuc/libXrdOuc.la \
#                         ../XrdSec/libXrdSec.la \
#                         ../XrdSys/libXrdSys.la

