#              $Id: GNUmakefile,v 1.28 2009/06/24 00:24:22 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdOuc -lXrd -lXrdNet -lXrdOuc -lXrdSys $(TYPELIBSF)

LIBDEPS = $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrd.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdOlbAdmin.cc     \
        XrdOlbCache.cc     \
        XrdOlbConfig.cc    \
        XrdOlbManager.cc   \
        XrdOlbManList.cc   \
        XrdOlbManTree.cc   \
        XrdOlbMeter.cc     \
        XrdOlbPList.cc     \
        XrdOlbPrepare.cc   \
        XrdOlbPrepArgs.cc  \
        XrdOlbProtocol.cc  \
        XrdOlbReq.cc       \
        XrdOlbRRQ.cc       \
        XrdOlbRTable.cc    \
        XrdOlbServer.cc    \
        XrdOlbState.cc     \
        XrdOlbXmiReq.cc
  
OBJECTS = \
        $(OBJDIR)/XrdOlbAdmin.o     \
        $(OBJDIR)/XrdOlbCache.o     \
        $(OBJDIR)/XrdOlbConfig.o    \
        $(OBJDIR)/XrdOlbManager.o   \
        $(OBJDIR)/XrdOlbManList.o   \
        $(OBJDIR)/XrdOlbManTree.o   \
        $(OBJDIR)/XrdOlbMeter.o     \
        $(OBJDIR)/XrdOlbPList.o     \
        $(OBJDIR)/XrdOlbPrepare.o   \
        $(OBJDIR)/XrdOlbPrepArgs.o  \
        $(OBJDIR)/XrdOlbProtocol.o  \
        $(OBJDIR)/XrdOlbReq.o       \
        $(OBJDIR)/XrdOlbRRQ.o       \
        $(OBJDIR)/XrdOlbRTable.o    \
        $(OBJDIR)/XrdOlbServer.o    \
        $(OBJDIR)/XrdOlbState.o     \
        $(OBJDIR)/XrdOlbXmiReq.o

BINARY  = $(BINDIR)/olbd

TARGETS = $(BINARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XProtocol% ../XProtocol
vpath XrdNet%    ../XrdNet
vpath XrdOuc%    ../XrdOuc
vpath XrdSys%    ../XrdSys
vpath XrdXrootd% ../XrdXrootd
vpath Xrd%       ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdOlb done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(BINARY): $(OBJECTS) $(LIBDEPS)
	@echo Creating executable $(BINARY) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTS) $(BINLIBS) $(LIBS) -o $(BINARY)

$(OBJDIR)/XrdOlbAdmin.o:  XrdOlbAdmin.hh    XrdOlbAdmin.cc \
                          XrdNetSocket.hh \
                          XrdOlbConfig.hh   XrdOlbManager.hh XrdOlbPrepare.hh\
                          XrdOlbTrace.hh \
                          XrdOuca2x.hh      XrdSysError.hh   XrdOucName2Name.hh\
                          XrdSysPlatform.hh XrdSysPthread.hh XrdOucStream.hh
	@echo Compiling XrdOlbAdmin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbAdmin.o XrdOlbAdmin.cc

$(OBJDIR)/XrdOlbCache.o: XrdOlbCache.hh  XrdOlbCache.cc \
                         XrdJob.hh       XrdScheduler.hh \
                         XrdOlbPList.hh  XrdOlbRRQ.hh      XrdOlbTypes.hh \
                         XrdOucHash.hh   XrdOucHash.icc    XrdSysPthread.hh
	@echo Compiling XrdOlbCache.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbCache.o XrdOlbCache.cc

$(OBJDIR)/XrdOlbConfig.o: XrdOlbConfig.hh   XrdOlbConfig.cc \
                          XrdJob.hh         XrdScheduler.hh \
                          XrdNetDNS.hh      XrdNetLink.hh    XrdNetSecurity.hh \
                          XrdNetSocket.hh   XrdNetWork.hh \
                          XrdOlbCache.hh    XrdOlbManager.hh XrdOlbMeter.hh \
                          XrdOlbPList.hh    XrdOlbPrepare.hh XrdOlbRRQ.hh \
                          XrdOlbState.hh    XrdOlbTrace.hh   XrdOlbTypes.hh \
                          XrdOlbXmi.hh      XrdOlbXmiReq.hh  XrdOlbServer.hh \
                          XrdOuca2x.hh      XrdSysError.hh   XrdOucName2Name.hh\
                          XrdOucPList.hh    XrdOucExport.hh \
                          XrdSysPlatform.hh XrdSysPlugin.hh  XrdOucProg.hh \
                          XrdSysPthread.hh  XrdOucStream.hh  XrdSysTimer.hh \
                          XrdSysHeaders.hh \
                          XrdOucTList.hh    XrdOucUtils.hh   XrdOucEnv.hh \
                          ../XrdVersion.hh
	@echo Compiling XrdOlbConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbConfig.o XrdOlbConfig.cc

$(OBJDIR)/XrdOlbManager.o: XrdOlbManager.hh XrdOlbManager.cc \
                           XrdJob.hh        XrdScheduler.hh \
                           XrdNetWork.hh    XrdOlbAdmin.hh    XrdOlbManTree.hh \
                           XrdOlbCache.hh   XrdOlbConfig.hh   XrdOlbManager.hh \
                           XrdOlbManList.hh XrdOlbRTable.hh   XrdOlbServer.hh \
                           XrdOlbState.hh   XrdOlbTrace.hh    XrdOlbTypes.hh \
                           XrdOuca2x.hh     XrdSysPthread.hh
	@echo Compiling XrdOlbManager.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbManager.o XrdOlbManager.cc

$(OBJDIR)/XrdOlbManList.o: XrdOlbManList.cc   XrdOlbManList.hh \
                           XrdNetDNS.hh       XrdSysPlatform.hh  XrdSysPthread.hh
	@echo Compiling XrdOlbManList.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbManList.o XrdOlbManList.cc

$(OBJDIR)/XrdOlbManTree.o: XrdOlbManTree.cc   XrdOlbManTree.hh \
                           XrdSysPthread.hh   XrdOlbTrace.hh  \
                           XrdOlbManager.hh   XrdOlbServer.hh
	@echo Compiling XrdOlbManTree.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbManTree.o XrdOlbManTree.cc

$(OBJDIR)/XrdOlbMeter.o: XrdOlbMeter.hh  XrdOlbMeter.cc  \
                         XrdOlbConfig.hh XrdOlbManager.hh  XrdOlbTrace.hh \
                         XrdSysError.hh  XrdSysPlatform.hh XrdSysPthread.hh \
                         XrdOucStream.hh XrdOucTList.hh
	@echo Compiling XrdOlbMeter.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbMeter.o XrdOlbMeter.cc

$(OBJDIR)/XrdOlbPList.o: XrdOlbPList.hh XrdOlbPList.cc \
                         XrdOlbTypes.hh XrdSysPthread.hh
	@echo Compiling XrdOlbPList.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbPList.o XrdOlbPList.cc

$(OBJDIR)/XrdOlbPrepare.o: XrdOlbPrepare.hh  XrdOlbPrepare.cc  XrdOlbPrepArgs.hh \
                           XrdJob.hh         XrdScheduler.hh \
                           XrdOlbServer.hh   XrdOlbTrace.hh  \
                           XrdSysError.hh    XrdOucHash.hh     XrdOucHash.icc \
                           XrdSysPthread.hh  XrdOucStream.hh   XrdOucTList.hh \
                           XrdOucEnv.hh      XrdOucMsubs.hh
	@echo Compiling XrdOlbPrepare.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbPrepare.o XrdOlbPrepare.cc

$(OBJDIR)/XrdOlbPrepArgs.o: XrdOlbPrepArgs.hh  XrdOlbPrepArgs.cc  \
                            XrdJob.hh          XrdScheduler.hh \
                            XrdOlbConfig.hh    XrdOlbServer.hh  \
                            XrdSysPthread.hh
	@echo Compiling XrdOlbPrepArgs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbPrepArgs.o XrdOlbPrepArgs.cc

$(OBJDIR)/XrdOlbProtocol.o: XrdOlbProtocol.hh  XrdOlbProtocol.cc  \
                            XrdInet.hh         XrdLink.hh       XrdProtocol.hh   \
                            XrdNetLink.hh      XrdNetPeer.hh \
                            XrdOlbConfig.hh    XrdOlbManager.hh XrdOlbTrace.hh \
                            XrdSysError.hh     XrdSysHeaders.hh \
                            XrdSysLogger.hh    XrdSysPthread.hh
	@echo Compiling XrdOlbProtocol.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbProtocol.o XrdOlbProtocol.cc

$(OBJDIR)/XrdOlbReq.o: XrdOlbReq.hh     XrdOlbReq.cc   \
                       XrdSysError.hh   XrdSysPthread.hh  XrdOlbRRQ.hh \
                       XrdOlbRTable.hh  XrdOlbServer.hh   XrdOlbTrace.hh \
                       XrdXrootdStat.icc
	@echo Compiling XrdOlbReq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbReq.o XrdOlbReq.cc

$(OBJDIR)/XrdOlbRRQ.o: XrdOlbRRQ.hh     XrdOlbRRQ.cc   \
                       XrdOucDLlist.hh  XrdSysError.hh  XrdSysPthread.hh \
                       XrdSysTimer.hh   XProtocol.hh \
                       XrdOlbManager.hh XrdOlbRTable.hh XrdOlbServer.hh \
                       XrdOlbTrace.hh   XrdOlbTypes.hh
	@echo Compiling XrdOlbRRQ.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbRRQ.o XrdOlbRRQ.cc

$(OBJDIR)/XrdOlbRTable.o: XrdOlbRTable.hh     XrdOlbRTable.cc   \
                          XrdSysPthread.hh    XrdOlbServer.hh
	@echo Compiling XrdOlbRTable.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbRTable.o XrdOlbRTable.cc

$(OBJDIR)/XrdOlbServer.o: XrdOlbServer.hh   XrdOlbServer.cc \
                          XrdJob.hh         XrdScheduler.hh \
                          XrdNetLink.hh  \
                          XrdOlbCache.hh    XrdOlbConfig.hh    XrdOlbManager.hh\
                          XrdOlbManList.hh  XrdOlbMeter.hh     XrdOlbPrepare.hh\
                          XrdOlbReq.hh      XrdOlbRRQ.hh \
                          XrdOlbServer.hh   XrdOlbState.hh \
                          XrdOlbTrace.hh    XrdOlbTypes.hh     XrdOlbXmi.hh \
                          XrdOuca2x.hh      XrdOucName2Name.hh XrdSysPlatform.hh\
                          XrdOucProg.hh     XrdSysPthread.hh   XrdOucStream.hh
	@echo Compiling XrdOlbServer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbServer.o XrdOlbServer.cc

$(OBJDIR)/XrdOlbState.o: XrdOlbState.hh   XrdOlbState.cc \
                         XrdOlbManager.hh XrdOlbTypes.hh XrdSysPthread.hh
	@echo Compiling XrdOlbState.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbState.o XrdOlbState.cc

$(OBJDIR)/XrdOlbXmiReq.o: XrdOlbXmiReq.hh   XrdOlbXmiReq.cc \
                          XrdOlbReq.hh      XrdOlbTrace.hh  \
                          XrdOlbXmi.hh \
                          XrdSysError.hh    XrdOucTrace.hh  XrdSysPthread.hh
	@echo Compiling XrdOlbXmiReq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOlbXmiReq.o XrdOlbXmiReq.cc
