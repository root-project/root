#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

# global settings
AM_CPPFLAGS = -I$(top_srcdir)/src
AM_LDFLAGS=-static

#############################################################################
## extra directory definitions for install tree
xrootdxrdmonincdir = $(includedir)/xrootd/XrdMon

#############################################################################
## deliverables

bin_PROGRAMS = \
	   xrdmonCollector \
	   xrdmonAdmin

noinst_LTLIBRARIES = libXrdMonCommon.la \
                     libXrdMonDecoder.la \
                     libXrdMonCollector.la \
                     libXrdMonDummySender.la

# these targets will only be built with "make check"
check_PROGRAMS = xrdmonDecoder \
	   xrdmonDummySender \
	   xrdmonDumpPackets \
	   xrdmonDebugPacket \
	   xrdmonConvertTime \
           xrdmonCout2file \
	   testParser

xrootdxrdmoninc_HEADERS = \
XrdMonArgParser.hh         XrdMonDecOnePacket.hh      XrdMonSndDebug.hh \
XrdMonArgParserConvert.hh  XrdMonDecPacketDecoder.hh  XrdMonSndDictEntry.hh \
XrdMonBufferedOutput.hh    XrdMonDecPreProcess.hh     XrdMonSndDummyXrootd.hh \
XrdMonCommon.hh            XrdMonDecSink.hh           XrdMonSndPacket.hh \
XrdMonCtrAdmin.hh          XrdMonDecStageInfo.hh      XrdMonSndStageEntry.hh \
XrdMonCtrArchiver.hh       XrdMonDecTraceInfo.hh      XrdMonSndTraceCache.hh \
XrdMonCtrBuffer.hh         XrdMonDecUserInfo.hh       XrdMonSndTraceEntry.hh \
XrdMonCtrCollector.hh      XrdMonErrors.hh            XrdMonSndTransmitter.hh \
XrdMonCtrDebug.hh          XrdMonException.hh         XrdMonTimer.hh \
XrdMonCtrPacket.hh         XrdMonHeader.hh            XrdMonTypes.hh \
XrdMonCtrWriter.hh         XrdMonSenderInfo.hh        XrdMonUtils.hh \
XrdMonDecArgParser.hh      XrdMonSndAdminEntry.hh \
XrdMonDecDictInfo.hh       XrdMonSndCoder.hh \
XrdMonArgParser.icc  XrdMonTimer.icc


#############################################################################
# Build rules

libXrdMonCommon_la_SOURCES = \
	    XrdMonArgParser.cc XrdMonArgParser.hh XrdMonArgParser.icc         \
	    XrdMonCommon.cc XrdMonCommon.hh             \
	    XrdMonException.cc XrdMonException.hh          \
	    XrdMonHeader.cc XrdMonHeader.hh             \
	    XrdMonSenderInfo.cc XrdMonSenderInfo.hh         \
	    XrdMonTimer.cc XrdMonTimer.hh XrdMonTimer.icc              \
	    XrdMonUtils.cc XrdMonUtils.hh XrdMonErrors.hh

libXrdMonDecoder_la_SOURCES = \
	    XrdMonDecArgParser.cc XrdMonDecArgParser.hh       \
	    XrdMonDecDictInfo.cc XrdMonDecDictInfo.hh        \
	    XrdMonDecOnePacket.cc XrdMonDecOnePacket.hh       \
	    XrdMonDecPacketDecoder.cc XrdMonDecPacketDecoder.hh   \
	    XrdMonDecPreProcess.cc XrdMonDecPreProcess.hh      \
	    XrdMonBufferedOutput.cc XrdMonBufferedOutput.hh \
	    XrdMonDecSink.cc XrdMonDecSink.hh            \
            XrdMonDecStageInfo.cc XrdMonDecStageInfo.hh \
	    XrdMonDecTraceInfo.cc XrdMonDecTraceInfo.hh       \
	    XrdMonDecUserInfo.cc XrdMonDecUserInfo.hh
libXrdMonDecoder_la_LIBADD = $(LIBSOCKET) $(LIBNSL)

libXrdMonCollector_la_SOURCES = \
	    XrdMonCtrAdmin.cc XrdMonCtrAdmin.hh           \
	    XrdMonCtrArchiver.cc XrdMonCtrArchiver.hh        \
	    XrdMonCtrBuffer.cc XrdMonCtrBuffer.hh          \
	    XrdMonCtrDebug.cc XrdMonCtrDebug.hh           \
	    XrdMonCtrCollector.cc XrdMonCtrPacket.hh       \
	    XrdMonCtrWriter.cc XrdMonCtrWriter.hh
libXrdMonCollector_la_LIBADD = $(LIBSOCKET) $(LIBNSL)

libXrdMonDummySender_la_SOURCES = \
	    XrdMonSndCoder.cc XrdMonSndCoder.hh           \
	    XrdMonSndDebug.cc XrdMonSndDebug.hh           \
	    XrdMonSndDictEntry.cc XrdMonSndDictEntry.hh       \
            XrdMonSndStageEntry.cc XrdMonSndStageEntry.hh \
	    XrdMonSndDummyXrootd.cc XrdMonSndDummyXrootd.hh     \
	    XrdMonSndPacket.cc  XrdMonSndPacket.hh         \
	    XrdMonSndTraceCache.cc XrdMonSndTraceCache.hh      \
	    XrdMonSndTraceEntry.cc XrdMonSndTraceEntry.hh      \
	    XrdMonSndTransmitter.cc XrdMonSndTransmitter.hh
libXrdMonDummySender_la_LIBADD = $(LIBSOCKET) $(LIBNSL)

xrdmonCollector_SOURCES = XrdMonCtrMainApp.cc XrdMonCtrCollector.hh XrdMonArgParserConvert.hh \
                          XrdMonTypes.hh
xrdmonCollector_LDADD = libXrdMonCollector.la \
                        libXrdMonDecoder.la \
                        libXrdMonCommon.la \
                        ../XrdOuc/libXrdOuc.la \
                        ../XrdSys/libXrdSys.la

xrdmonDecoder_SOURCES = XrdMonDecMainApp.cc
xrdmonDecoder_LDADD =   libXrdMonDecoder.la \
                        libXrdMonCommon.la \
                        ../XrdOuc/libXrdOuc.la \
                        ../XrdSys/libXrdSys.la

xrdmonDummySender_SOURCES = XrdMonSndDummyXrootdApp.cc
xrdmonDummySender_LDADD = libXrdMonDummySender.la \
                          libXrdMonCommon.la \
                          -lresolv

xrdmonAdmin_SOURCES = XrdMonSndAdminApp.cc XrdMonSndAdminEntry.hh
xrdmonAdmin_LDADD = libXrdMonDummySender.la \
                    libXrdMonCommon.la \
                    -lresolv

xrdmonDumpPackets_SOURCES = XrdMonDumpPacketsApp.cc
xrdmonDumpPackets_LDADD = libXrdMonDecoder.la \
                          libXrdMonCommon.la

xrdmonDebugPacket_SOURCES = XrdMonDebugPacketApp.cc
xrdmonDebugPacket_LDADD = libXrdMonDecoder.la \
                          libXrdMonCommon.la

xrdmonConvertTime_SOURCES = XrdMonConvertTimestampApp.cc
xrdmonConvertTime_LDADD = libXrdMonCommon.la

testParser_SOURCES = testArgParser.cc
testParser_LDADD = libXrdMonCommon.la

xrdmonCout2file_SOURCES = XrdMonCout2FileApp.cc 
xrdmonCout2file_LDADD = libXrdMonDecoder.la \
                        ../XrdOuc/libXrdOuc.la \
                        libXrdMonCommon.la