#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdCnsConfig.cc \
        XrdCnsDaemon.cc \
        XrdCnsInventory.cc  \
        XrdCnsLog.cc  \
        XrdCnsLogClient.cc  \
        XrdCnsLogFile.cc  \
        XrdCnsLogRec.cc  \
        XrdCnsLogServer.cc  \
        XrdCnsMain.cc \
        XrdCnsSsi.cc \
        XrdCnsSsiCfg.cc \
        XrdCnsSsiMain.cc \
        XrdCnsXref.cc

OBJCNSD  = \
        $(OBJDIR)/XrdCnsConfig.o \
        $(OBJDIR)/XrdCnsDaemon.o \
        $(OBJDIR)/XrdCnsInventory.o  \
        $(OBJDIR)/XrdCnsLogClient.o  \
        $(OBJDIR)/XrdCnsLogFile.o  \
        $(OBJDIR)/XrdCnsLogServer.o  \
        $(OBJDIR)/XrdCnsMain.o

OBJCOM = \
        $(OBJDIR)/XrdCnsLog.o  \
        $(OBJDIR)/XrdCnsLogRec.o \
        $(OBJDIR)/XrdCnsXref.o

OBJSSI = \
        $(OBJDIR)/XrdCnsSsi.o \
        $(OBJDIR)/XrdCnsSsiCfg.o \
        $(OBJDIR)/XrdCnsSsiMain.o

OBJECTS = $(OBJCNSD) $(OBJCOM) $(OBJSSI)

LIBDEP   = \
        $(LIBDIR)/libXrdOss.a \
        $(LIBDIR)/libXrdFrm.a \
        $(LIBDIR)/libXrdClient.a \
        $(LIBDIR)/libXrdOuc.a \
        $(LIBDIR)/libXrdSys.a \
        $(LIBDIR)/libXrdNet.a \
        $(LIBDIR)/libXrdNetUtil.a


MORELIBS = -L$(LIBDIR)

TARGCNSD = $(BINDIR)/XrdCnsd

TARGSSI  = $(BINDIR)/cns_ssi

TARGETS  = $(TARGCNSD) $(TARGSSI)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdClient% ../XrdClient
vpath XrdNet%    ../XrdNet
vpath XrdOss%    ../XrdOss
vpath XrdOuc%    ../XrdOuc
vpath XrdSys%    ../XrdSys
vpath Xrd%       ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdCns done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGCNSD): $(OBJCNSD) $(OBJCOM) $(LIBDEP)
	@echo Creating executable $(BINDIR)/XrdCnsd
	$(ECHO)$(LD) $(LDOP) $(CFLAGS) $(OBJCNSD) $(OBJCOM) $(LIBDEP) $(MORELIBS) $(LIBS) -o $(BINDIR)/XrdCnsd

$(TARGSSI): $(OBJSSI) $(OBJCOM) $(LIBDEP)
	@echo Creating executable $(BINDIR)/cns_ssi
	$(ECHO)$(LD) $(LDOP) $(CFLAGS) $(OBJSSI) $(OBJCOM) $(LIBDEP) $(MORELIBS) $(LIBS) -o $(BINDIR)/cns_ssi

$(OBJDIR)/XrdCnsConfig.o:  XrdCnsConfig.hh    XrdCnsConfig.cc \
                           XrdCnsDaemon.hh    XrdCnsLogRec.hh \
                           XrdCnsLogServer.hh XrdCnsXref.hh \
                           XrdClientConst.hh  XrdClientEnv.hh \
                           XrdNetOpts.hh      XrdNetSocket.hh \
                           XrdNetDNS.hh       XrdTrace.hh \
                           XrdOuca2x.hh \
                           XrdOucArgs.hh      XrdOucName2Name.hh \
                           XrdOucStream.hh    XrdOucTList.hh \
                           XrdOucTokenizer.hh XrdOucUtils.hh \
                           XrdSysError.hh     XrdSysLogger.hh\
                           XrdSysHeaders.hh   XrdSysPlugin.hh \
                           XrdSysTimer.hh
	@echo Compiling XrdCnsConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsConfig.o XrdCnsConfig.cc

$(OBJDIR)/XrdCnsDaemon.o: XrdCnsDaemon.hh    XrdCnsDaemon.cc \
                          XrdCnsLogRec.hh    XrdTrace.hh \
                          XrdSysError.hh     XrdSysLogger.hh \
                          XrdSysHeaders.hh   XrdSysPthread.hh
	@echo Compiling XrdCnsDaemon.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsDaemon.o XrdCnsDaemon.cc

$(OBJDIR)/XrdCnsLog.o: XrdCnsLog.hh    XrdCnsLog.cc   XrdCnsLogRec.hh \
                       XrdOucNSWalk.hh XrdOucTList.hh \
                       XrdNetDNS.hh \
                       XrdSysError.hh
	@echo Compiling XrdCnsLog.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsLog.o XrdCnsLog.cc

$(OBJDIR)/XrdCnsInventory.o: XrdCnsInventory.hh  XrdCnsInventory.cc  \
                             XrdCnsLogFile.hh    XrdCnsLogRec.hh \
                             XrdCnsConfig.hh     XrdCnsXref.hh \
                             XrdOssPath.hh       XrdOssSpace.hh \
                             XrdOucTList.hh      XrdOucNSWalk.hh \
                             XrdSysError.hh
	@echo Compiling XrdCnsInventory.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsInventory.o XrdCnsInventory.cc

$(OBJDIR)/XrdCnsLogClient.o: XrdCnsLogClient.hh  XrdCnsLogClient.cc  \
                             XrdCnsLogFile.hh    XrdCnsLogRec.hh \
                             XrdCnsConfig.hh     XrdCnsLog.hh \
                             XrdCnsInventory.hh \
                             XrdClient.hh        XrdClientAdmin.hh \
                             XrdNetDNS.hh \
                             XrdOucTList.hh \
                             XrdOucNSWalk.hh     XrdOucUtils.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysTimer.hh      XrdTrace.hh
	@echo Compiling XrdCnsLogClient.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsLogClient.o XrdCnsLogClient.cc

$(OBJDIR)/XrdCnsLogFile.o: XrdCnsLogFile.hh  XrdCnsLogFile.cc  \
                           XrdCnsLogRec.hh   XrdSysPlatform.hh \
                           XrdSysError.hh  XrdSysPthread.hh
	@echo Compiling XrdCnsLogFile.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsLogFile.o XrdCnsLogFile.cc

$(OBJDIR)/XrdCnsLogRec.o: XrdCnsLogRec.hh  XrdCnsLogRec.cc  \
                          XrdSysPthread.hh XrdSysTimer.hh
	@echo Compiling XrdCnsLogRec.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsLogRec.o XrdCnsLogRec.cc

$(OBJDIR)/XrdCnsLogServer.o: XrdCnsLogServer.hh  XrdCnsLogServer.cc  \
                             XrdCnsLogClient.hh  XrdCnsLogFile.hh \
                             XrdCnsLogRec.hh     XrdCnsConfig.hh \
                             XrdOucTList.hh \
                             XrdTrace.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysPlatform.hh
	@echo Compiling XrdCnsLogServer.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsLogServer.o XrdCnsLogServer.cc

$(OBJDIR)/XrdCnsMain.o:    XrdCnsDaemon.hh    XrdCnsMain.cc \
                           XrdCnsConfig.hh \
                           XrdTrace.hh        XrdOucStream.hh \
                           XrdSysError.hh     XrdSysHeaders.hh XrdSysLogger.hh  \
                           XrdSysPlatform.hh  XrdSysPthread.hh XrdSysTimer.hh
	@echo Compiling XrdCnsMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsMain.o XrdCnsMain.cc

$(OBJDIR)/XrdCnsSsi.o:     XrdCnsSsi.cc       XrdCnsSsi.hh \
                           XrdCnsSsiCfg.hh    XrdCnsXref.hh \
                           XrdCnsLog.hh       XrdCnsLogRec.hh \
                           XrdOucHash.hh      XrdOucUtils.hh \
                           XrdOucNSWalk.hh    XrdOucSxeq.hh \
                           XrdOucStream.hh    XrdOucTList.hh \
                           XrdSysError.hh     XrdSysHeaders.hh
	@echo Compiling XrdCnsSsi.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsSsi.o XrdCnsSsi.cc

$(OBJDIR)/XrdCnsSsiCfg.o:  XrdCnsSsiCfg.cc    XrdCnsSsiCfg.hh \
                           XrdCnsLog.hh       XrdCnsSsiSay.hh \
                           XrdOuca2x.hh       XrdOucArgs.hh \
                           XrdOucTList.hh     XrdOucUtils.hh \
                           XrdSysError.hh     XrdSysLogger.hh
	@echo Compiling XrdCnsSsiCfg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsSsiCfg.o XrdCnsSsiCfg.cc

$(OBJDIR)/XrdCnsSsiMain.o: XrdCnsSsiMain.cc   XrdCnsSsiCfg.hh \
                           XrdCnsSsi.hh       XrdCnsSsiSay.hh \
                           XrdOucTList.hh \
                           XrdSysError.hh     XrdSysLogger.hh \
                           XrdSysPthread.hh   XrdSysTimer.hh
	@echo Compiling XrdCnsSsiMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsSsiMain.o XrdCnsSsiMain.cc

$(OBJDIR)/XrdCnsXref.o:    XrdCnsXref.cc   XrdCnsXref.hh \
                           XrdOucHash.hh \
                           XrdSysPthread.hh
	@echo Compiling XrdCnsXref.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsXref.o XrdCnsXref.cc
