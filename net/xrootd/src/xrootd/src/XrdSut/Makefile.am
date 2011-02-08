#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

xrootdsutincdir = $(includedir)/xrootd/XrdSut

xrootdsutinc_HEADERS =  XrdSutAux.hh  XrdSutBucket.hh  XrdSutBuckList.hh  \
			XrdSutBuffer.hh  XrdSutCache.hh  XrdSutPFEntry.hh  \
			XrdSutPFile.hh  XrdSutRndm.hh  XrdSutTrace.hh

lib_LTLIBRARIES = libXrdSut.la

INCLUDES = -I$(top_srcdir)/src

libXrdSut_la_SOURCES = \
        XrdSutAux.cc XrdSutAux.hh XrdSutTrace.hh    \
        XrdSutCache.cc XrdSutCache.hh \
        XrdSutBucket.cc XrdSutBucket.hh \
        XrdSutBuckList.cc XrdSutBuckList.hh \
        XrdSutBuffer.cc XrdSutBuffer.hh \
        XrdSutPFile.cc XrdSutPFile.hh\
        XrdSutPFEntry.cc XrdSutPFEntry.hh \
        XrdSutRndm.cc XrdSutRndm.hh
