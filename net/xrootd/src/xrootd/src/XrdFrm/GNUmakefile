#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdFrm -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys -ldl \
            $(TYPELIBSF)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdFrmAdmin.cc     \
        XrdFrmAdminAudit.cc\
        XrdFrmAdminConvert.cc\
        XrdFrmAdminFiles.cc \
        XrdFrmAdminFind.cc \
        XrdFrmAdminMain.cc \
        XrdFrmAdminQuery.cc \
        XrdFrmAdminUnlink.cc\
        XrdFrmCID.cc       \
        XrdFrmConfig.cc    \
        XrdFrmFiles.cc     \
        XrdFrmMonitor.cc   \
        XrdFrmMigrate.cc   \
        XrdFrmProxy.cc     \
        XrdFrmPurge.cc     \
        XrdFrmPurgMain.cc  \
        XrdFrmReqAgent.cc  \
        XrdFrmReqBoss.cc   \
        XrdFrmReqFile.cc   \
        XrdFrmTrace.cc     \
        XrdFrmTransfer.cc  \
        XrdFrmTSort.cc     \
        XrdFrmUtils.cc     \
        XrdFrmXfrAgent.cc  \
        XrdFrmXfrDaemon.cc \
        XrdFrmXfrMain.cc   \
        XrdFrmXfrQueue.cc

OBJLIB  = \
        $(OBJDIR)/XrdFrmCID.o       \
        $(OBJDIR)/XrdFrmFiles.o     \
        $(OBJDIR)/XrdFrmMonitor.o  \
        $(OBJDIR)/XrdFrmProxy.o     \
        $(OBJDIR)/XrdFrmReqAgent.o  \
        $(OBJDIR)/XrdFrmReqFile.o   \
        $(OBJDIR)/XrdFrmTrace.o     \
        $(OBJDIR)/XrdFrmTSort.o     \
        $(OBJDIR)/XrdFrmUtils.o

OBJADMIN= \
        $(OBJDIR)/XrdFrmConfig.o    \
        $(OBJDIR)/XrdFrmAdminMain.o \
        $(OBJDIR)/XrdFrmAdmin.o     \
        $(OBJDIR)/XrdFrmAdminAudit.o\
        $(OBJDIR)/XrdFrmAdminConvert.o\
        $(OBJDIR)/XrdFrmAdminFiles.o\
        $(OBJDIR)/XrdFrmAdminFind.o \
        $(OBJDIR)/XrdFrmAdminQuery.o\
        $(OBJDIR)/XrdFrmAdminUnlink.o

OBJPURG = \
        $(OBJDIR)/XrdFrmConfig.o    \
        $(OBJDIR)/XrdFrmPurgMain.o  \
        $(OBJDIR)/XrdFrmPurge.o

OBJXFR  = \
        $(OBJDIR)/XrdFrmConfig.o    \
        $(OBJDIR)/XrdFrmMigrate.o  \
        $(OBJDIR)/XrdFrmReqBoss.o   \
        $(OBJDIR)/XrdFrmTransfer.o \
        $(OBJDIR)/XrdFrmXfrAgent.o \
        $(OBJDIR)/XrdFrmXfrDaemon.o \
        $(OBJDIR)/XrdFrmXfrMain.o  \
        $(OBJDIR)/XrdFrmXfrQueue.o

OBJECTS = $(OBJLIB) $(OBJADMIN) $(OBJPURG) $(OBJXFR)

OBJADDS = \
        $(LIBDIR)/libXrdOss.a

OBJDEPS = \
          $(LIBDIR)/libXrdFrm.a \
          $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdNetUtil.a \
          $(LIBDIR)/libXrdOss.a \
          $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a

LIBRARY = $(LIBDIR)/libXrdFrm.a

BINADMIN= $(BINDIR)/frm_admin
BINPURGD= $(BINDIR)/frm_purged
BINXFRD = $(BINDIR)/frm_xfrd
BINAGNT = $(BINDIR)/frm_xfragent

TARGETS = $(LIBRARY) $(BINADMIN) $(BINPURGD) $(BINXFRD) $(BINAGNT)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdCms% ../XrdCms
vpath XrdFrm% ../XrdFrm
vpath XrdNet% ../XrdNet
vpath XrdOss% ../XrdOss
vpath XrdOuc% ../XrdOuc
vpath XrdSys% ../XrdSys
vpath XrdXrootd% ../XrdXrootd
vpath Xrd%    ../Xrd
vpath XP%     ../XProtocol

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdFrm done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJLIB)
	@echo Creating archive $(LIBRARY)
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJLIB)
	$(ECHO)ranlib $(LIBRARY)

$(BINADMIN): $(OBJLIB) $(OBJADMIN)          $(OBJDEPS)
	@echo Creating executable $(BINADMIN)
	$(ECHO)$(LD) $(LDOP) $(OBJADMIN) $(OBJADDS) $(LIBREADLINE) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINADMIN)

$(BINPURGD): $(OBJLIB) $(OBJPURG) $(OBJADDS) $(OBJDEPS)
	@echo Creating executable $(BINPURGD)
	$(ECHO)$(LD) $(LDOP) $(OBJPURG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINPURGD)

$(BINXFRD): $(OBJLIB) $(OBJXFR) $(OBJADDS) $(OBJDEPS)
	@echo Creating executables $(BINXFRD)
	$(ECHO)$(LD) $(LDOP) $(OBJXFR) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINXFRD)

$(BINAGNT): $(BINXFRD)
	@echo Creating executables $(BINAGNT)
	$(ECHO)rm -f $(BINAGNT)
	$(ECHO)ln $(BINXFRD) $(BINAGNT)

$(OBJDIR)/XrdFrmAdmin.o:  XrdFrmAdmin.hh    XrdFrmAdmin.cc  XrdFrmTrace.hh \
                          XrdFrmConfig.hh   XrdFrmProxy.hh  XrdFrmUtils.hh  \
                          XrdOss.hh         XrdOucExport.hh \
                          XrdOuca2x.hh      XrdOucArgs.hh   \
                          XrdOucTList.hh    XrdOucTokenizer.hh \
                          XrdSysError.hh
	@echo Compiling XrdFrmAdmin.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdmin.o XrdFrmAdmin.cc

$(OBJDIR)/XrdFrmAdminAudit.o:  XrdFrmAdmin.hh    XrdFrmAdminAudit.cc \
                               XrdFrmConfig.hh   XrdFrmFiles.hh  \
                               XrdFrmTrace.hh    XrdFrmUtils.hh \
                               XrdOssPath.hh     XrdOssSpace.hh \
                               XrdOucNSWalk.hh   XrdOucTList.hh \
                               XrdSysError.hh    XrdSysTimer.hh
	@echo Compiling XrdFrmAdminAudit.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminAudit.o XrdFrmAdminAudit.cc

$(OBJDIR)/XrdFrmAdminConvert.o: XrdFrmAdmin.hh    XrdFrmAdminConvert.cc \
                                XrdFrmConfig.hh   XrdFrmFiles.hh  \
                                XrdFrmTrace.hh    XrdFrmUtils.hh \
                                XrdFrmXAttr.hh    XrdOss.hh \
                                XrdOssPath.hh     XrdOucArgs.hh \
                                XrdOucNSWalk.hh   XrdOucPList.hh \
                                XrdOucTList.hh    XrdOucXAttr.hh  \
                                XrdSysError.hh
	@echo Compiling XrdFrmAdminConvert.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminConvert.o XrdFrmAdminConvert.cc

$(OBJDIR)/XrdFrmAdminFiles.o: XrdFrmAdmin.hh    XrdFrmAdminFiles.cc \
                              XrdFrmConfig.hh   XrdFrmFiles.hh  \
                              XrdFrmTrace.hh    XrdFrmXAttr.hh \
                              XrdOucXAttr.hh  \
                              XrdSysError.hh
	@echo Compiling XrdFrmAdminFiles.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminFiles.o XrdFrmAdminFiles.cc

$(OBJDIR)/XrdFrmAdminFind.o:  XrdFrmAdmin.hh    XrdFrmAdminFind.cc \
                              XrdFrmConfig.hh   XrdFrmFiles.hh  \
                              XrdFrmTrace.hh    XrdSysError.hh \
                              XrdOucArgs.hh     XrdOucNSWalk.hh
	@echo Compiling XrdFrmAdminFind.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminFind.o XrdFrmAdminFind.cc

$(OBJDIR)/XrdFrmAdminMain.o:  XrdFrmAdmin.hh     XrdFrmAdminMain.cc \
                              XrdFrmConfig.hh    XrdFrmTrace.hh \
                              XrdNetOpts.hh      XrdNetSocket.hh \
                              XrdOucTokenizer.hh \
                              XrdSysError.hh     XrdSysHeaders.hh \
                              XrdSysLogger.hh
	@echo Compiling XrdFrmAdminMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFREADLINE) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminMain.o XrdFrmAdminMain.cc

$(OBJDIR)/XrdFrmAdminQuery.o:  XrdFrmAdmin.hh    XrdFrmAdminQuery.cc \
                               XrdFrmConfig.hh   XrdFrmFiles.hh  \
                               XrdFrmRequest.hh  XrdFrmProxy.hh \
                               XrdFrmTrace.hh    XrdFrmUtils.hh \
                               XrdSysError.hh \
                               XrdOssPath.hh     XrdOssSpace.hh \
                               XrdOucArgs.hh     XrdOucNSWalk.hh \
                               XrdOucTList.hh
	@echo Compiling XrdFrmAdminQuery.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminQuery.o XrdFrmAdminQuery.cc

$(OBJDIR)/XrdFrmAdminUnlink.o: XrdFrmAdmin.hh    XrdFrmAdminUnlink.cc \
                               XrdFrmConfig.hh   XrdFrmTrace.hh \
                               XrdFrmUtils.hh    XrdSysError.hh \
                               XrdOss.hh         XrdOssPath.hh \
                               XrdOucNSWalk.hh   XrdNetCmsNotify.hh
	@echo Compiling XrdFrmAdminUnlink.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminUnlink.o XrdFrmAdminUnlink.cc

$(OBJDIR)/XrdFrmCID.o: XrdFrmCID.hh    XrdFrmCID.cc    XrdFrmTrace.hh \
                       XrdOucEnv.hh    XrdOucStream.hh \
                       XrdSysError.hh  XrdSysPlatform.hh
	@echo Compiling XrdFrmCID.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmCID.o XrdFrmCID.cc

$(OBJDIR)/XrdFrmConfig.o: XrdFrmConfig.hh  XrdFrmConfig.cc    XrdFrmTrace.hh \
                          XrdFrmMonitor.hh XrdNetCmsNotify.hh \
                          XrdInfo.hh       XrdNetDNS.hh       XrdOss.hh \
                          XrdOuca2x.hh     XrdOucEnv.hh       XrdOucExport.hh \
                          XrdOucUtils.hh   XrdOucName2Name.hh XrdOucMsubs.hh \
                          XrdOucPList.hh   XrdOucProg.hh \
                          XrdOucStream.hh  XrdOucTList.hh     XrdOssSpace.hh \
                          XrdSysError.hh   XrdSysHeaders.hh   XrdSysLogger.hh \
                          XrdSysTimer.hh   XrdSysPlugin.hh    XrdSysPthread.hh \
                          XrdSysPlatform.hh
	@echo Compiling XrdFrmConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmConfig.o XrdFrmConfig.cc

$(OBJDIR)/XrdFrmFiles.o:  XrdFrmFiles.hh    XrdFrmFiles.cc  XrdFrmTrace.hh \
                          XrdFrmConfig.hh   XrdFrmUtils.hh  XrdFrmXAttr.hh \
                          XrdOssPath.hh \
                          XrdOucHash.hh     XrdOucNSWalk.hh XrdOucXAttr.hh \
                          XrdSysError.hh    XrdSysPlatform.hh
	@echo Compiling XrdFrmFiles.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmFiles.o XrdFrmFiles.cc

$(OBJDIR)/XrdFrmMigrate.o: XrdFrmMigrate.cc     XrdFrmMigrate.hh \
                           XrdFrmConfig.hh      XrdFrmFiles.hh \
                           XrdFrmRequest.hh     XrdFrmTrace.hh \
                           XrdFrmTransfer.hh    XrdFrmXAttr.hh \
                           XrdFrmXfrQueue.hh \
                           XrdOss.hh            XrdOssPath.hh \
                           XrdOucTList.hh \
                           XrdOucNSWalk.hh      XrdOucUtils.hh \
                           XrdSysError.hh       XrdSysHeaders.hh \
                           XrdSysLogger.hh      XrdSysPlatform.hh \
                           XrdSysPthread.hh     XrdSysTimer.hh
	@echo Compiling XrdFrmMigrate.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmMigrate.o XrdFrmMigrate.cc

$(OBJDIR)/XrdFrmMonitor.o: XrdFrmMonitor.hh     XrdFrmMonitor.cc \
                           XrdFrmTrace.hh \
                           XrdNet.hh            XrdNetDNS.hh \
                           XrdNetPeer.hh \
                           XrdSysError.hh       XrdSysPlatform.hh \
                           XrdXrootdMonData.hh  XPtypes.hh
	@echo Compiling XrdFrmMonitor.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmMonitor.o XrdFrmMonitor.cc

$(OBJDIR)/XrdFrmProxy.o: XrdFrmProxy.hh       XrdFrmProxy.cc \
                         XrdFrmReqAgent.hh    XrdFrmRequest.hh \
                         XrdFrmTrace.hh       XrdFrmUtils.hh \
                         XrdOucEnv.hh         XrdOucStream.hh \
                         XrdOucUtils.hh \
                         XrdSysError.hh       XrdSysLogger.hh \
                         XrdSysPlatform.hh
	@echo Compiling XrdFrmProxy.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmProxy.o XrdFrmProxy.cc

$(OBJDIR)/XrdFrmPurge.o:  XrdFrmPurge.hh   XrdFrmPurge.cc    XrdFrmTrace.hh \
                          XrdFrmConfig.hh  XrdFrmFiles.hh    XrdFrmTSort.hh \
                          XrdFrmXAttr.hh \
                          XrdNetCmsNotify.hh \
                          XrdOss.hh        XrdOssPath.hh     XrdOssSpace.hh \
                          XrdOucNSWalk.hh  XrdOucUtils.hh \
                          XrdOucProg.hh    XrdOucStream.hh   XrdSysError.hh \
                          XrdSysPlatform.hh
	@echo Compiling XrdFrmPurge.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPurge.o XrdFrmPurge.cc

$(OBJDIR)/XrdFrmPurgMain.o: XrdFrmPurgMain.cc \
                            XrdFrmConfig.hh      XrdFrmPurge.hh \
                            XrdFrmTrace.hh \
                            XrdNetOpts.hh        XrdNetSocket.hh \
                            XrdOucUtils.cc \
                            XrdSysError.hh       XrdSysHeaders.hh \
                            XrdSysLogger.hh      XrdSysPthread.hh \
                            XrdSysTimer.hh
	@echo Compiling XrdFrmPurgMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPurgMain.o XrdFrmPurgMain.cc

$(OBJDIR)/XrdFrmReqAgent.o: XrdFrmReqAgent.hh XrdFrmReqAgent.cc \
                            XrdFrmReqFile.hh  XrdFrmRequest.hh \
                            XrdFrmTrace.hh    XrdFrmUtils.hh \
                            XrdNetMsg.hh  \
                            XrdSysError.hh    XrdSysHeaders.hh \
                            XrdSysPlatform.hh
	@echo Compiling XrdFrmReqAgent.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmReqAgent.o XrdFrmReqAgent.cc

$(OBJDIR)/XrdFrmReqBoss.o: XrdFrmReqBoss.hh  XrdFrmReqBoss.cc \
                           XrdFrmReqFile.hh  XrdFrmRequest.hh \
                           XrdFrmTrace.hh    XrdFrmUtils.hh \
                           XrdFrmXfrQueue.hh XrdFrmCID.hh \
                           XrdNetMsg.hh      XrdOucUtils.hh \
                           XrdSysHeaders.hh  XrdSysPthread.hh \
                           XrdSysError.hh
	@echo Compiling XrdFrmReqBoss.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmReqBoss.o XrdFrmReqBoss.cc

$(OBJDIR)/XrdFrmReqFile.o: XrdFrmReqFile.cc      XrdFrmReqFile.hh \
                           XrdFrmRequest.hh      XrdFrmCID.hh \
                           XrdFrmTrace.hh \
                           XrdSysError.hh        XrdSysPthread.hh \
                           XrdSysPlatform.hh
	@echo Compiling XrdFrmReqFile.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmReqFile.o XrdFrmReqFile.cc


$(OBJDIR)/XrdFrmTrace.o:  XrdFrmTrace.hh    XrdFrmTrace.cc
	@echo Compiling XrdFrmTrace.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmTrace.o XrdFrmTrace.cc

$(OBJDIR)/XrdFrmTransfer.o: XrdFrmTransfer.cc     XrdFrmTransfer.hh \
                            XrdFrmConfig.hh       XrdFrmReqFile.hh \
                            XrdFrmRequest.hh      XrdFrmTrace.hh \
                            XrdFrmXfrJob.hh       XrdFrmXfrQueue.hh \
                            XrdFrmMonitor.hh      XrdFrmCID.hh \
                            XrdFrmXAttr.hh \
                            XrdOss.hh \
                            XrdOucEnv.hh          XrdOucHash.hh \
                            XrdOucMsubs.hh        XrdOucProg.hh \
                            XrdOucSxeq.hh         XrdOucXAttr.hh \
                            XrdNetCmsNotify.hh \
                            XrdSysError.hh        XrdSysPlatform.hh \
                            XrdSysPthread.hh
	@echo Compiling XrdFrmTransfer.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmTransfer.o XrdFrmTransfer.cc


$(OBJDIR)/XrdFrmTSort.o:  XrdFrmTSort.hh    XrdFrmTSort.cc  XrdFrmFiles.hh
	@echo Compiling XrdFrmTSort.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmTSort.o XrdFrmTSort.cc

$(OBJDIR)/XrdFrmUtils.o:  XrdFrmUtils.hh    XrdFrmUtils.cc \
                          XrdFrmRequest.hh  XrdFrmTrace.hh   XrdFrmXLock.hh \
                          XrdFrmXAttr.hh \
                          XrdOucSxeq.hh     XrdOucUtils.hh   XrdOucXAttr.hh \
                          XrdSysError.hh    XrdSysHeaders.hh XrdSysPlatform.hh
	@echo Compiling XrdFrmUtils.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmUtils.o XrdFrmUtils.cc

$(OBJDIR)/XrdFrmXfrAgent.o: XrdFrmXfrAgent.hh XrdFrmXfrAgent.cc \
                            XrdFrmConfig.hh   XrdFrmReqAgent.hh \
                            XrdFrmReqFile.hh  XrdFrmRequest.hh \
                            XrdFrmTrace.hh    XrdFrmUtils.hh \
                            XrdOucStream.hh \
                            XrdSysError.hh    XrdSysPlatform.hh
	@echo Compiling XrdFrmXfrAgent.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmXfrAgent.o XrdFrmXfrAgent.cc

$(OBJDIR)/XrdFrmXfrDaemon.o: XrdFrmXfrDaemon.hh XrdFrmXfrDaemon.cc \
                            XrdFrmConfig.hh     XrdFrmMigrate.hh \
                            XrdFrmReqBoss.hh    XrdFrmRequest.hh \
                            XrdFrmTrace.hh      XrdFrmTransfer.hh \
                            XrdFrmUtils.hh \
                            XrdFrmXfrAgent.hh   XrdFrmXfrDaemon.hh \
                            XrdNetOpts.hh       XrdNetSocket.hh   \
                            XrdOucStream.hh \
                            XrdSysError.hh      XrdSysPlatform.hh \
                            XrdSysPthread.hh    XrdSysTimer.hh
	@echo Compiling XrdFrmXfrDaemon.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmXfrDaemon.o XrdFrmXfrDaemon.cc

$(OBJDIR)/XrdFrmXfrMain.o: XrdFrmXfrMain.cc \
                           XrdFrmConfig.hh      XrdFrmTrace.hh \
                           XrdFrmXfrAgent.hh    XrdFrmXfrDaemon.hh \
                           XrdSysError.hh       XrdSysHeaders.hh \
                           XrdSysLogger.hh      XrdSysPthread.hh
	@echo Compiling XrdFrmXfrMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmXfrMain.o XrdFrmXfrMain.cc

$(OBJDIR)/XrdFrmXfrQueue.o: XrdFrmXfrQueue.cc     XrdFrmXfrQueue.hh \
                            XrdFrmConfig.hh       XrdFrmReqFile.hh \
                            XrdFrmRequest.hh      XrdFrmTrace.hh \
                            XrdFrmXfrJob.hh \
                            XrdNetMsg.hh \
                            XrdOucHash.hh         XrdOucTList.hh \
                            XrdSysError.hh        XrdSysPlatform.hh \
                            XrdSysPthread.hh      XrdSysTimer.hh
	@echo Compiling XrdFrmXfrQueue.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmXfrQueue.o XrdFrmXfrQueue.cc
