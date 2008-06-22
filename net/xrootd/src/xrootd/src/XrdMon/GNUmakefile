#              $Id: GNUmakefile,v 1.27 2008/05/12 23:34:31 becla Exp $

#-----------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                       #
#-----------------------------------------------------------------------------#

include ../GNUmake.env

#-----------------------------------------------------------------------------#
#                             C o m p o n e n t s                             #
#-----------------------------------------------------------------------------#

OBJECTS_LIB_COMMON =                          \
        $(OBJDIR)/XrdMonArgParser.o           \
        $(OBJDIR)/XrdMonCommon.o              \
        $(OBJDIR)/XrdMonException.o           \
        $(OBJDIR)/XrdMonHeader.o              \
	$(OBJDIR)/XrdMonSenderInfo.o          \
        $(OBJDIR)/XrdMonTimer.o               \
        $(OBJDIR)/XrdMonUtils.o

OBJECTS_LIB_CTR =                             \
	$(OBJDIR)/XrdMonCtrAdmin.o            \
	$(OBJDIR)/XrdMonCtrArchiver.o         \
	$(OBJDIR)/XrdMonCtrBuffer.o           \
	$(OBJDIR)/XrdMonCtrDebug.o            \
	$(OBJDIR)/XrdMonCtrCollector.o        \
	$(OBJDIR)/XrdMonCtrWriter.o

OBJECTS_LIB_DEC =                             \
	$(OBJDIR)/XrdMonDecArgParser.o        \
        $(OBJDIR)/XrdMonDecDictInfo.o         \
	$(OBJDIR)/XrdMonDecOnePacket.o        \
	$(OBJDIR)/XrdMonDecMainApp.o          \
        $(OBJDIR)/XrdMonDecPacketDecoder.o    \
	$(OBJDIR)/XrdMonDecPreProcess.o       \
	$(OBJDIR)/XrdMonBufferedOutput.o      \
	$(OBJDIR)/XrdMonDecSink.o             \
        $(OBJDIR)/XrdMonDecStageInfo.o        \
	$(OBJDIR)/XrdMonDecTraceInfo.o        \
        $(OBJDIR)/XrdMonDecUserInfo.o         \

OBJECTS_LIB_SND =                             \
	$(OBJDIR)/XrdMonSndCoder.o            \
	$(OBJDIR)/XrdMonSndDebug.o            \
	$(OBJDIR)/XrdMonSndDictEntry.o        \
	$(OBJDIR)/XrdMonSndStageEntry.o       \
	$(OBJDIR)/XrdMonSndDummyXrootd.o      \
	$(OBJDIR)/XrdMonSndPacket.o           \
	$(OBJDIR)/XrdMonSndTraceCache.o       \
	$(OBJDIR)/XrdMonSndTraceEntry.o       \
	$(OBJDIR)/XrdMonSndTransmitter.o

OBJECTS_BIN_CTR = $(OBJDIR)/XrdMonCtrMainApp.o
OBJECTS_BIN_DEC = $(OBJDIR)/XrdMonDecMainApp.o
OBJECTS_BIN_ADM = $(OBJDIR)/XrdMonSndAdminApp.o
OBJECTS_BIN_SND = $(OBJDIR)/XrdMonSndDummyXrootdApp.o
OBJECTS_BIN_FSD = $(OBJDIR)/XrdMonSndFileSenderApp.o
OBJECTS_BIN_DMP = $(OBJDIR)/XrdMonDumpPacketsApp.o
OBJECTS_BIN_DBP = $(OBJDIR)/XrdMonDebugPacketApp.o
OBJECTS_BIN_CTM = $(OBJDIR)/XrdMonConvertTimestampApp.o
OBJECTS_BIN_C2F = $(OBJDIR)/XrdMonCout2FileApp.o
OBJECTS_BIN_TPR = $(OBJDIR)/testArgParser.o

LIBRARY_OUC    = $(LIBDIR)/libXrdOuc.a

LIBRARY_COMMON = $(LIBDIR)/libXrdMonCommon.a
LIBRARY_DEC    = $(LIBDIR)/libXrdMonDecoder.a
LIBRARY_CTR    = $(LIBDIR)/libXrdMonCollector.a
LIBRARY_SND    = $(LIBDIR)/libXrdMonDummySender.a

LIBRARIES      = $(LIBRARY_COMMON) $(LIBRARY_CTR) \
                 $(LIBRARY_DEC) $(LIBRARY_SND) 

COLLECTOR      = $(BINDIR)/xrdmonCollector
DECODER        = $(BINDIR)/xrdmonDecoder
SENDER         = $(BINDIR)/xrdmonDummySender
FSENDER        = $(BINDIR)/xrdmonFileSender
ADMIN          = $(BINDIR)/xrdmonAdmin
DUMPPACKETS    = $(BINDIR)/xrdmonDumpPackets
DEBUGPACKET    = $(BINDIR)/xrdmonDebugPacket
CONVTIME       = $(BINDIR)/xrdmonConvertTime
TESTPARSER     = $(BINDIR)/testParser
COUT2FILE      = $(BINDIR)/xrdmonCout2file

BINARIES       = $(COLLECTOR)        \
                 $(ADMIN)            \
                 $(COUT2FILE)       \
                 $(SENDER)          \
#                 $(FSENDER)         \
#                 $(DECODER)         \
#                 $(DUMPPACKETS)     \
#                 $(DEBUGPACKET)     \
#                 $(CONVTIME)        \
#                 $(TESTPARSER)

TARGETS        = $(LIBRARIES) $(BINARIES)

#-----------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           #
#-----------------------------------------------------------------------------#

vpath XrdMon% ../XrdMon

#-----------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                          #
#-----------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdMon done.

#-----------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                           #
#-----------------------------------------------------------------------------#

$(LIBRARY_COMMON): $(OBJECTS_LIB_COMMON)
	@echo Creating archive $(LIBRARY_COMMON) 
	$(ECHO)rm -f $(LIBRARY_COMMON)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		$(CC) -xar -o $(LIBRARY_COMMON) $(OBJECTS_LIB_COMMON); \
	else \
		ar -rc $(LIBRARY_COMMON) $(OBJECTS_LIB_COMMON); \
		ranlib $(LIBRARY_COMMON); \
	fi

$(LIBRARY_CTR):	$(OBJECTS_LIB_CTR)
	@echo Creating archive $(LIBRARY_CTR) 
	$(ECHO)rm -f $(LIBRARY_CTR)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		$(CC) -xar -o $(LIBRARY_CTR) $(OBJECTS_LIB_CTR); \
	else \
		ar -rc $(LIBRARY_CTR) $(OBJECTS_LIB_CTR); \
		ranlib $(LIBRARY_CTR); \
	fi

$(LIBRARY_DEC):	$(OBJECTS_LIB_DEC)
	@echo Creating archive $(LIBRARY_DEC) 
	$(ECHO)rm -f $(LIBRARY_DEC)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		$(CC) -xar -o $(LIBRARY_DEC) $(OBJECTS_LIB_DEC); \
	else \
		ar -rc $(LIBRARY_DEC) $(OBJECTS_LIB_DEC); \
		ranlib $(LIBRARY_DEC); \
	fi

$(LIBRARY_SND): $(OBJECTS_LIB_SND)
	@echo Creating archive $(LIBRARY_SND) 
	$(ECHO)rm -f $(LIBRARY_SND)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		$(CC) -xar -o $(LIBRARY_SND) $(OBJECTS_LIB_SND); \
	else \
		ar -rc $(LIBRARY_SND) $(OBJECTS_LIB_SND); \
		ranlib $(LIBRARY_SND); \
	fi
$(COLLECTOR):	$(OBJECTS_BIN_CTR) $(LIBRARY_CTR) \
	$(LIBRARY_DEC) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -o $@ $^ -L$(LIBDIR) -lXrdSys; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -o $@ $^ -L$(LIBDIR) -lXrdSys; \
	fi
$(DECODER):	$(OBJECTS_BIN_DEC) $(LIBRARY_DEC) $(LIBRARY_OUC) \
	$(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)$(CC) $(CFLAGS) $(LDOP) $(LIBS) -o $@ $^

$(SENDER):	$(OBJECTS_BIN_SND) $(LIBRARY_SND) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -lresolv -o $@ $^; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -lresolv -o $@ $^; \
	fi

$(FSENDER):	$(OBJECTS_BIN_FSD) $(LIBRARY_SND) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -lresolv -o $@ $^; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -lresolv -o $@ $^; \
	fi

$(ADMIN):	$(OBJECTS_BIN_ADM) $(LIBRARY_SND) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -lresolv -o $@ $^; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -lresolv -o $@ $^; \
	fi

$(COUT2FILE):	$(OBJECTS_BIN_C2F) $(LIBRARY_DEC) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -lresolv -o $@ $^; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -lresolv -o $@ $^; \
	fi

$(DUMPPACKETS):	$(OBJECTS_BIN_DMP) $(LIBRARY_DEC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)$(CC) $(CFLAGS) $(LDOP) $(LIBS) -o $@ $^

$(DEBUGPACKET):	$(OBJECTS_BIN_DBP) $(LIBRARY_DEC) $(LIBRARY_OUC) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)$(CC) $(CFLAGS) $(LDOP) $(LIBS) -o $@ $^

$(CONVTIME):	$(OBJECTS_BIN_CTM) $(LIBRARY_COMMON) 
	@echo Creating executable $(@F)
	$(ECHO)$(CC) $(CFLAGS) $(LDOP) $(LIBS) -o $@ $^

$(TESTPARSER):	$(OBJECTS_BIN_TPR) $(LIBRARY_COMMON)
	@echo Creating executable $(@F)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		if [ "x$(SUNCACHE)" != "x" ]; then \
			$(CC) -xar -o lib_temp.a $(OBJDIR)$(SUNCACHE)/*/*.o; \
			$(CC) $(CFLAGS) $(LDOP) lib_temp.a $(LIBS) -lresolv -o $@ $^; \
			rm lib_temp.a; \
		fi; \
	else \
		$(CC) $(CFLAGS) $(LDOP) $(LIBS) -lresolv -o $@ $^; \
	fi

$(OBJDIR)/%.o : %.cc
	@echo Compiling $(<F)
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) $< -o $@

