#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.5 2007/11/01 09:38:19 dfeich Exp $
## Checked in by $Author: dfeich $
#######################################################################

xrootdxrdmonincdir = $(includedir)/xrootd/XrdMon

xrootdxrdmoninc_HEADERS = \
XrdMonArgParser.hh         XrdMonDecDictInfo.hh       XrdMonSndCoder.hh \
XrdMonArgParserConvert.hh  XrdMonDecOnePacket.hh      XrdMonSndDebug.hh \
XrdMonBufferedOutput.hh    XrdMonDecPacketDecoder.hh  XrdMonSndDictEntry.hh \
XrdMonCommon.hh            XrdMonDecPreProcess.hh     XrdMonSndDummyXrootd.hh \
XrdMonCtrAdmin.hh          XrdMonDecSink.hh           XrdMonSndPacket.hh \
XrdMonCtrArchiver.hh       XrdMonDecTraceInfo.hh      XrdMonSndTraceCache.hh \
XrdMonCtrBuffer.hh         XrdMonDecUserInfo.hh       XrdMonSndTraceEntry.hh \
XrdMonCtrCollector.hh      XrdMonErrors.hh            XrdMonSndTransmitter.hh \
XrdMonCtrDebug.hh          XrdMonException.hh         XrdMonTimer.hh \
XrdMonCtrPacket.hh         XrdMonHeader.hh            XrdMonTypes.hh \
XrdMonCtrWriter.hh         XrdMonSenderInfo.hh        XrdMonUtils.hh \
XrdMonDecArgParser.hh      XrdMonSndAdminEntry.hh \
XrdMonArgParser.icc  XrdMonTimer.icc

noinst_LTLIBRARIES = libXrdMonCommon.la \
                     libXrdMonDecoder.la \
                     libXrdMonCollector.la \
                     libXrdMonDummySender.la

bin_PROGRAMS = \
	   xrdmonCollector \
	   xrdmonAdmin

# note: these targets were disabled in the original Makefiles
#       I add them as 'check' targets
check_PROGRAMS = xrdmonDecoder \
	   xrdmonDummySender \
	   xrdmonDumpPackets \
	   xrdmonDebugPacket \
	   xrdmonConvertTime \
	   testParser

INCLUDES = -I$(top_srcdir)/src

AM_LDFLAGS=-static

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
