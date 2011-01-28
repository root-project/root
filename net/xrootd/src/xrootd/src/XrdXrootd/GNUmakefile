#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdCms -lXrd -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys \
            $(TYPELIBSF)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdXrootdAdmin.cc      \
        XrdXrootdAio.cc        \
        XrdXrootdCallBack.cc   \
        XrdXrootdConfig.cc     \
        XrdXrootdFile.cc       \
        XrdXrootdFileLock1.cc \
        XrdXrootdJob.cc     \
        XrdXrootdLoadLib.cc \
        XrdXrootdMonitor.cc   \
        XrdXrootdPio.cc       \
        XrdXrootdPrepare.cc   \
        XrdXrootdProtocol.cc   \
        XrdXrootdResponse.cc   \
        XrdXrootdStats.cc      \
        XrdXrootdXeq.cc        \
        XrdXrootdXeqAio.cc
  
OBJECTS = \
        $(OBJDIR)/XrdXrootdAdmin.o      \
        $(OBJDIR)/XrdXrootdAio.o        \
        $(OBJDIR)/XrdXrootdCallBack.o   \
        $(OBJDIR)/XrdXrootdConfig.o     \
        $(OBJDIR)/XrdXrootdFile.o       \
        $(OBJDIR)/XrdXrootdFileLock1.o \
        $(OBJDIR)/XrdXrootdJob.o     \
        $(OBJDIR)/XrdXrootdLoadLib.o \
        $(OBJDIR)/XrdXrootdMonitor.o \
        $(OBJDIR)/XrdXrootdPio.o     \
        $(OBJDIR)/XrdXrootdPrepare.o \
        $(OBJDIR)/XrdXrootdProtocol.o   \
        $(OBJDIR)/XrdXrootdResponse.o   \
        $(OBJDIR)/XrdXrootdStats.o      \
        $(OBJDIR)/XrdXrootdXeq.o        \
        $(OBJDIR)/XrdXrootdXeqAio.o

OBJFS   = \
          $(OBJDIR)/XrdOfsFS.o  \
          $(LIBDIR)/libXrdOfs.a \
          $(LIBDIR)/libXrdOss.a \
          $(LIBDIR)/libXrdFrm.a \
          $(LIBDIR)/libXrdAcc.a

OBJDEPS = \
          $(LIBDIR)/libXrd.a \
          $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdNetUtil.a \
          $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdCms.a \
          $(OBJFS)

TARGETS  = $(BINDIR)/xrootd

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdSfs%    ../XrdSfs
vpath XrdSec%    ../XrdSec
vpath XrdOuc%    ../XrdOuc
vpath XrdNet%    ../XrdNet
vpath XrdSys%    ../XrdSys
vpath XrdXrootd% ../XrdXrdXrootd
vpath Xrd%       ../Xrd
vpath XP%        ../XProtocol

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdXrootd done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(OBJDEPS)
	@echo Creating executable $(TARGETS) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTS) $(OBJFS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(TARGETS)

$(OBJDIR)/XrdXrootdAdmin.o: XrdXrootdAdmin.hh      XrdXrootdAdmin.cc \
                            XrdXrootdTrace.hh      XrdXrootdProtocol.hh \
                            XrdXrootdJob.hh \
                            XPtypes.hh             XProtocol.hh \
                            XrdLink.hh             XrdNetSocket.hh \
                            XrdSysError.hh         XrdLinkMatch.hh \
                            XrdSysPlatform.hh      XrdSysPthread.hh \
                            XrdOucStream.hh        XrdOucTList.hh
	@echo Compiling XrdXrootdAdmin.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdAdmin.o XrdXrootdAdmin.cc

$(OBJDIR)/XrdXrootdAio.o: XrdXrootdAio.hh      XPtypes.hh           \
                          XrdBuffer.hh         XrdLink.hh           \
                          XrdScheduler.hh \
                          XrdSysError.hh       XrdSysPthread.hh     \
                          XrdSfsAio.hh         XrdSfsInterface.hh   \
                          XrdXrootdFile.hh     XrdXrootdProtocol.hh \
                          XrdXrootdResponse.hh XrdXrootdStats.hh    \
                          XrdXrootdTrace.hh    XrdXrootdAio.cc
	@echo Compiling XrdXrootdAio.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdAio.o XrdXrootdAio.cc

$(OBJDIR)/XrdXrootdCallBack.o: XrdXrootdCallBack.hh   XrdXrootdCallBack.cc \
                          XPtypes.hh           XProtocol.hh \
                          XrdScheduler.hh \
                          XrdSysError.hh       XrdOucErrInfo.hh \
                          XrdSysPthread.hh     \
                          XrdSfsInterface.hh   \
                          XrdXrootdProtocol.hh \
                          XrdXrootdReqID.hh    XrdXrootdResponse.hh \
                          XrdXrootdStats.hh    XrdXrootdTrace.hh
	@echo Compiling XrdXrootdCallBack.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdCallBack.o XrdXrootdCallBack.cc

$(OBJDIR)/XrdXrootdConfig.o: XrdXrootdProtocol.hh XrdXrootdFile.hh \
                             XrdXrootdFileLock.hh XrdXrootdFileLock1.hh \
                             XrdXrootdPrepare.hh  XrdXrootdXPath.hh \
                             XrdXrootdStats.hh    XrdXrootdTrace.hh \
                             XrdXrootdMonitor.hh  XrdXrootdAio.hh \
                             XrdXrootdAdmin.hh    XrdProtocol.hh \
                             XrdXrootdJob.hh      XrdXrootdCallBack.hh \
                             XrdNetDNS.hh         XrdNetSocket.hh  XrdNetOpts.hh \
                             XrdBuffer.hh         XProtocol.hh \
                             XrdOuca2x.hh         XrdSysError.hh  \
                             XrdSysLogger.hh      XrdSysHeaders.hh \
                             XrdOucReqID.hh \
                             XrdOucTokenizer.hh   XrdOucTrace.hh \
                             XrdOucProg.hh        XrdXrootdConfig.cc \
                             XrdOucStream.hh      XrdOucEnv.hh
	@echo Compiling XrdXrootdConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdConfig.o XrdXrootdConfig.cc

$(OBJDIR)/XrdXrootdFile.o:   XrdSfsInterface.hh   XrdXrootdFile.hh \
                             XrdXrootdFileLock.hh \
                             XrdXrootdFile.cc     XrdXrootdTrace.hh \
                             XrdSysError.hh       XrdOucTrace.hh \
                             XrdSysPthread.hh     XPtypes.hh
	@echo Compiling XrdXrootdFile.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdFile.o XrdXrootdFile.cc

$(OBJDIR)/XrdXrootdFileLock1.o: XrdXrootdFile.hh XrdXrootdFileLock1.hh \
                                 XrdOucHash.hh   XrdOucHash.icc XrdSysPthread.hh \
                                 XrdXrootdFileLock1.cc
	@echo Compiling XrdXrootdFileLock1.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdFileLock1.o XrdXrootdFileLock1.cc

$(OBJDIR)/XrdXrootdJob.o:    XrdXrootdJob.hh        XrdXrootdJob.cc \
                             XrdLink.hh             XrdScheduler.hh \
                             XrdOucProg.hh          XrdSysPthread.hh \
                             XrdSysPlatform.hh      XrdOucStream.hh \
                             XrdOucTable.hh         XrdOucTList.hh \
                             XrdXrootdResponse.hh   XrdXrootdTrace.hh \
                             XProtocol.hh           XPtypes.hh
	@echo Compiling XrdXrootdJob.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdJob.o XrdXrootdJob.cc

$(OBJDIR)/XrdXrootdLoadLib.o: XrdSfsInterface.hh XrdSecInterface.hh XrdSysError.hh \
                              XrdXrootdLoadLib.cc  XrdOucEnv.hh
	@echo Compiling XrdXrootdLoadLib.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdLoadLib.o XrdXrootdLoadLib.cc

$(OBJDIR)/XrdXrootdMonitor.o: XrdSysError.hh      \
                              XrdSysPthread.hh    XrdSysPlatform.hh \
                              XrdNet.hh           XrdNetPeer.hh     XrdNetDNS.hh \
                              XrdScheduler.hh     XrdJob.hh        \
                              XPtypes.hh          XrdXrootdMonitor.hh \
                              XrdXrootdMonData.hh XrdXrootdMonitor.cc  \
                              XrdXrootdTrace.hh
	@echo Compiling XrdXrootdMonitor.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdMonitor.o XrdXrootdMonitor.cc

$(OBJDIR)/XrdXrootdPio.o:  XrdXrootdPio.hh  XrdXrootdPio.cc \
                           XrdSysPthread.hh XPtypes.hh
	@echo Compiling XrdXrootdPio.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdPio.o XrdXrootdPio.cc

$(OBJDIR)/XrdXrootdPrepare.o: XrdSysError.hh      XrdOucTList.hh   \
                              XrdJob.hh           XrdScheduler.hh  \
                              XrdXrootdTrace.hh   XrdSysPlatform.hh \
                              XrdXrootdPrepare.hh XrdXrootdPrepare.cc
	@echo Compiling XrdXrootdPrepare.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdPrepare.o XrdXrootdPrepare.cc

$(OBJDIR)/XrdXrootdProtocol.o: XrdXrootdFile.hh       XrdXrootdFileLock.hh \
                               XrdXrootdFileLock1.hh  XrdXrootdXPath.hh \
                               XrdXrootdProtocol.hh   XrdXrootdResponse.hh \
                               XrdXrootdStats.hh      XrdXrootdTrace.hh    \
                               XrdXrootdAio.hh        XProtocol.hh         \
                               XrdBuffer.hh           XrdLink.hh           \
                               XrdObject.hh           XrdObject.icc        \
                               XrdXrootdReqID.hh      XProtocol.hh \
                               XrdSfsInterface.hh     XrdSecInterface.hh \
                               XrdSecEntity.hh \
                               XrdSysError.hh         XrdSysPthread.hh   \
                               XrdSysTimer.hh         XrdOucTrace.hh \
                               XrdXrootdPio.hh        \
                               XrdXrootdProtocol.hh   XrdXrootdProtocol.cc
	@echo Compiling XrdXrootdProtocol.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdProtocol.o XrdXrootdProtocol.cc

$(OBJDIR)/XrdXrootdResponse.o: XrdLink.hh           XrdXrootdResponse.hh  \
                               XrdXrootdReqID.hh    XrdXrootdTrace.hh \
                               XProtocol.hh         XrdSysError.hh \
                               XrdOucTrace.hh       XrdXrootdResponse.cc \
                               ../XrdVersion.hh
	@echo Compiling XrdXrootdResponse.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdResponse.o XrdXrootdResponse.cc

$(OBJDIR)/XrdXrootdStats.o: XrdXrootdResponse.hh XrdStats.hh \
                            XrdSfsInterface.hh   XrdSysPthread.hh \
                            XrdXrootdStats.cc
	@echo Compiling XrdXrootdStats.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdStats.o XrdXrootdStats.cc

$(OBJDIR)/XrdXrootdXeq.o: XrdXrootdFile.hh      XrdXrootdFileLock.hh  \
                          XrdXrootdProtocol.hh  XrdXrootdStats.hh     \
                          XrdXrootdTrace.hh     XProtocol.hh       \
                          XrdXrootdPrepare.hh   XrdXrootdXPath.hh  \
                          XrdXrootdMonitor.hh   XrdXrootdMonData.hh \
                          XrdXrootdAio.hh       XrdXrootdReqID.hh \
                          XrdXrootdPio.hh       XrdXrootdStat.icc \
                          XrdBuffer.hh          XrdLink.hh            \
                          XrdSfsInterface.hh    XrdSecInterface.hh \
                          XrdSecEntity.hh       XrdSysTimer.hh \
                          XrdSysError.hh        XrdOucTrace.hh  \
                          XrdSysPlatform.hh     XrdOucTList.hh \
                          XrdOucReqID.hh        XrdOucStream.hh \
                          XrdXrootdJob.hh       XrdXrootdXeq.cc
	@echo Compiling XrdXrootdXeq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdXeq.o XrdXrootdXeq.cc

$(OBJDIR)/XrdXrootdXeqAio.o: XrdXrootdXeqAio.cc \
                             XrdBuffer.hh         XrdLink.hh       \
                             XrdSysError.hh       XrdOucErrInfo.hh \
                             XrdSfsInterface.hh   XrdXrootdCallBack.hh \
                             XrdXrootdAio.hh      XrdXrootdFile.hh \
                             XrdXrootdProtocol.hh XrdXrootdTrace.hh
	@echo Compiling XrdXrootdXeqAio.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdXrootdXeqAio.o XrdXrootdXeqAio.cc
