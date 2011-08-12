#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdOuca2x.cc         \
        XrdOucArgs.cc        \
        XrdOucCRC.cc         \
        XrdOucEnv.cc         \
        XrdOucExport.cc      \
        XrdOucHashVal.cc     \
        XrdOucMsubs.cc       \
        XrdOucName2Name.cc   \
        XrdOucNList.cc       \
        XrdOucNSWalk.cc      \
        XrdOucProg.cc        \
        XrdOucPup.cc         \
        XrdOucReqID.cc       \
        XrdOucStream.cc      \
        XrdOucString.cc      \
        XrdOucSxeq.cc        \
        XrdOucTokenizer.cc   \
        XrdOucTrace.cc       \
        XrdOucUtils.cc
ifneq ($(BUILDBJR),)
SOURCES += XrdOucBonjour.cc XrdOucFactoryBonjour.cc
endif

OBJECTS = \
        $(OBJDIR)/XrdOuca2x.o          \
        $(OBJDIR)/XrdOucArgs.o         \
        $(OBJDIR)/XrdOucCRC.o          \
        $(OBJDIR)/XrdOucEnv.o          \
        $(OBJDIR)/XrdOucExport.o       \
        $(OBJDIR)/XrdOucHashVal.o      \
        $(OBJDIR)/XrdOucMsubs.o        \
        $(OBJDIR)/XrdOucName2Name.o    \
        $(OBJDIR)/XrdOucNList.o        \
        $(OBJDIR)/XrdOucNSWalk.o       \
        $(OBJDIR)/XrdOucProg.o         \
        $(OBJDIR)/XrdOucPup.o          \
        $(OBJDIR)/XrdOucReqID.o        \
        $(OBJDIR)/XrdOucStream.o       \
        $(OBJDIR)/XrdOucString.o       \
        $(OBJDIR)/XrdOucSxeq.o         \
        $(OBJDIR)/XrdOucTokenizer.o    \
        $(OBJDIR)/XrdOucTrace.o        \
        $(OBJDIR)/XrdOucUtils.o        \
        $(OBJDIR)/XrdOucCRC.o
ifneq ($(BUILDBJR),)
OBJECTS +=  $(OBJDIR)/XrdOucBonjour.o  $(OBJDIR)/XrdOucFactoryBonjour.o
endif


LIBRARY = $(LIBDIR)/libXrdOuc.a

TARGETS = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOuc% ../XrdOuc
vpath XrdSys% ../XrdSys
vpath Xrd% ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdOuc done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJECTS)
	$(ECHO)ranlib $(LIBRARY)

$(OBJDIR)/XrdOuca2x.o: XrdOuca2x.cc XrdOuca2x.hh XrdSysError.hh
	@echo Compiling XrdOuca2x.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOuca2x.cc -o $(OBJDIR)/XrdOuca2x.o

$(OBJDIR)/XrdOucArgs.o: XrdOucArgs.cc  XrdOucArgs.hh XrdOucTokenizer.hh \
                        XrdSysError.hh
	@echo Compiling XrdOucArgs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucArgs.cc -o $(OBJDIR)/XrdOucArgs.o

$(OBJDIR)/XrdOucCRC.o: XrdOucCRC.cc XrdOucCRC.hh
	@echo Compiling XrdOucCRC.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucCRC.cc -o $(OBJDIR)/XrdOucCRC.o

$(OBJDIR)/XrdOucEnv.o: XrdOucEnv.cc XrdOucEnv.hh XrdOucHash.hh
	@echo Compiling XrdOucEnv.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucEnv.cc -o $(OBJDIR)/XrdOucEnv.o

$(OBJDIR)/XrdOucExport.o: XrdOucExport.cc XrdOucExport.hh \
                          XrdOucStream.hh XrdOucPList.hh  XrdSysError.hh \
                          XrdSysPlatform.hh
	@echo Compiling XrdOucExport.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucExport.cc -o $(OBJDIR)/XrdOucExport.o

$(OBJDIR)/XrdOucHashVal.o: XrdOucHashVal.cc
	@echo Compiling XrdOucHashVal.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucHashVal.cc -o $(OBJDIR)/XrdOucHashVal.o

$(OBJDIR)/XrdOucMsubs.o: XrdOucMsubs.cc XrdOucMsubs.hh \
                         XrdOucEnv.hh   XrdOucName2Name.hh
	@echo Compiling XrdOucMsubs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucMsubs.cc -o $(OBJDIR)/XrdOucMsubs.o

$(OBJDIR)/XrdOucName2Name.o: XrdOucName2Name.cc XrdOucName2Name.hh \
                             XrdSysError.hh XrdSysPlatform.hh
	@echo Compiling XrdOucName2Name.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucName2Name.cc -o $(OBJDIR)/XrdOucName2Name.o

$(OBJDIR)/XrdOucNList.o: XrdOucNList.cc XrdOucNList.hh XrdSysPthread.hh
	@echo Compiling XrdOucNList.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucNList.cc -o $(OBJDIR)/XrdOucNList.o

$(OBJDIR)/XrdOucNSWalk.o: XrdOucNSWalk.cc XrdOucNSWalk.hh XrdOucTList.hh \
                          XrdSysError.hh  XrdSysPlatform.hh
	@echo Compiling XrdOucNSWalk.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucNSWalk.cc -o $(OBJDIR)/XrdOucNSWalk.o

$(OBJDIR)/XrdOucProg.o: XrdOucProg.cc  XrdOucProg.hh \
                        XrdSysError.hh XrdOucStream.hh XrdSysPthread.hh
	@echo Compiling XrdOucProg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucProg.cc -o $(OBJDIR)/XrdOucProg.o

$(OBJDIR)/XrdOucPup.o: XrdOucPup.cc  XrdOucPup.hh XrdSysHeaders.hh
	@echo Compiling XrdOucPup.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucPup.cc -o $(OBJDIR)/XrdOucPup.o


$(OBJDIR)/XrdOucReqID.o: XrdOucReqID.cc XrdOucReqID.hh \
                         XrdOucCRC.hh   XrdSysPthread.hh
	@echo Compiling XrdOucReqID.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucReqID.cc -o $(OBJDIR)/XrdOucReqID.o

$(OBJDIR)/XrdOucStream.o: XrdOucStream.cc XrdOucStream.hh XrdSysError.hh \
                          XrdOucUtils.hh  XrdSysHeaders.hh \
                          XrdSysLogger.hh XrdSysPlatform.hh
	@echo Compiling XrdOucStream.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucStream.cc -o $(OBJDIR)/XrdOucStream.o

$(OBJDIR)/XrdOucString.o: XrdOucString.cc XrdOucString.hh XrdSysHeaders.hh
	@echo Compiling XrdOucString.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucString.cc -o $(OBJDIR)/XrdOucString.o

$(OBJDIR)/XrdOucSxeq.o: XrdOucSxeq.cc XrdOucSxeq.hh XrdSysPlatform.hh
	@echo Compiling XrdOucSxeq.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucSxeq.cc -o $(OBJDIR)/XrdOucSxeq.o

$(OBJDIR)/XrdOucTokenizer.o: XrdOucTokenizer.cc XrdOucTokenizer.hh  
	@echo Compiling XrdOucTokenizer.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucTokenizer.cc -o $(OBJDIR)/XrdOucTokenizer.o

$(OBJDIR)/XrdOucTrace.o: XrdSysError.hh XrdOucTrace.hh XrdOucTrace.cc \
                         XrdSysHeaders.hh
	@echo Compiling XrdOucTrace.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucTrace.cc -o $(OBJDIR)/XrdOucTrace.o

$(OBJDIR)/XrdOucUtils.o: XrdOucUtils.cc XrdOucUtils.hh \
                      XrdSysError.hh XrdOucStream.hh XrdNetDNS.hh
	@echo Compiling XrdOucUtils.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdOucUtils.cc -o $(OBJDIR)/XrdOucUtils.o

$(OBJDIR)/XrdOucBonjour.o: XrdOucBonjour.cc XrdOucBonjour.hh XrdSysError.hh \
                           XrdSysHeaders.hh XrdOucString.hh XrdSysPthread.hh \
                           XrdOucFactoryBonjour.hh \
                           XrdProtLoad.hh
	@echo Compiling XrdOucBonjour.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(BJRFLAGS) $(CFHASLIBDNSSD) $(CFHASLIBAVAHI) $(INCLUDE) XrdOucBonjour.cc -o $(OBJDIR)/XrdOucBonjour.o

$(OBJDIR)/XrdOucFactoryBonjour.o: XrdOucBonjour.hh XrdSysError.hh \
                                XrdSysHeaders.hh XrdOucString.hh XrdSysPthread.hh \
                                XrdOucFactoryBonjour.hh XrdOucFactoryBonjour.cc XrdConfig.hh \
                                XrdInet.hh XrdProtLoad.hh
	@echo Compiling XrdOucFactoryBonjour.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(BJRFLAGS) $(CFHASLIBDNSSD) $(INCLUDE) XrdOucFactoryBonjour.cc -o $(OBJDIR)/XrdOucFactoryBonjour.o
