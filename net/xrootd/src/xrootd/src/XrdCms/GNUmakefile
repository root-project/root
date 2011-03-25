#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdOss -lXrdFrm -lXrdNet -lXrdOuc -lXrdNetUtil \
                      -lXrd -lXrdSys $(TYPELIBSF)

LIBDEPS =  $(LIBDIR)/libXrdFrm.a     $(LIBDIR)/libXrdNet.a \
           $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdOss.a \
           $(LIBDIR)/libXrdOuc.a     $(LIBDIR)/libXrdSys.a $(LIBDIR)/libXrd.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SRCCLNT = \
        XrdCmsClientConfig.cc\
        XrdCmsClientMan.cc   \
        XrdCmsClientMsg.cc   \
        XrdCmsFinder.cc      \
        XrdCmsResp.cc        \

SRCBOTH = \
        XrdCmsLogin.cc       \
        XrdCmsParser.cc      \
        XrdCmsRRData.cc      \
        XrdCmsSecurity.cc    \
        XrdCmsTalk.cc

SRCSRVR = \
        XrdCmsAdmin.cc       \
        XrdCmsBaseFS.cc      \
        XrdCmsCache.cc       \
        XrdCmsCluster.cc     \
        XrdCmsConfig.cc      \
        XrdCmsJob.cc         \
        XrdCmsKey.cc         \
        XrdCmsManList.cc     \
        XrdCmsManTree.cc     \
        XrdCmsManager.cc     \
        XrdCmsMeter.cc       \
        XrdCmsNash.cc        \
        XrdCmsNode.cc        \
        XrdCmsPList.cc       \
        XrdCmsPrepArgs.cc    \
        XrdCmsPrepare.cc     \
        XrdCmsProtocol.cc    \
        XrdCmsReq.cc         \
        XrdCmsRouting.Cc     \
        XrdCmsRRQ.cc         \
        XrdCmsRTable.cc      \
        XrdCmsState.cc       \
        XrdCmsSupervisor.cc  \
        XrdCmsXmiReq.cc

SOURCES = $(SRCCLNT) $(SRCBOTH) $(SRCSRVR)
  
OBJCLNT = \
        $(OBJDIR)/XrdCmsClientConfig.o\
        $(OBJDIR)/XrdCmsClientMan.o   \
        $(OBJDIR)/XrdCmsClientMsg.o   \
        $(OBJDIR)/XrdCmsFinder.o      \
        $(OBJDIR)/XrdCmsResp.o

OBJBOTH = \
        $(OBJDIR)/XrdCmsLogin.o       \
        $(OBJDIR)/XrdCmsParser.o      \
        $(OBJDIR)/XrdCmsRRData.o      \
        $(OBJDIR)/XrdCmsSecurity.o    \
        $(OBJDIR)/XrdCmsTalk.o

OBJSRVR = \
        $(OBJDIR)/XrdCmsAdmin.o       \
        $(OBJDIR)/XrdCmsBaseFS.o      \
        $(OBJDIR)/XrdCmsCache.o       \
        $(OBJDIR)/XrdCmsCluster.o     \
        $(OBJDIR)/XrdCmsConfig.o      \
        $(OBJDIR)/XrdCmsJob.o         \
        $(OBJDIR)/XrdCmsKey.o         \
        $(OBJDIR)/XrdCmsManList.o     \
        $(OBJDIR)/XrdCmsManTree.o     \
        $(OBJDIR)/XrdCmsManager.o     \
        $(OBJDIR)/XrdCmsMeter.o       \
        $(OBJDIR)/XrdCmsNash.o        \
        $(OBJDIR)/XrdCmsNode.o        \
        $(OBJDIR)/XrdCmsPList.o       \
        $(OBJDIR)/XrdCmsPrepArgs.o    \
        $(OBJDIR)/XrdCmsPrepare.o     \
        $(OBJDIR)/XrdCmsProtocol.o    \
        $(OBJDIR)/XrdCmsReq.o         \
        $(OBJDIR)/XrdCmsRouting.o     \
        $(OBJDIR)/XrdCmsRRQ.o         \
        $(OBJDIR)/XrdCmsRTable.o      \
        $(OBJDIR)/XrdCmsState.o       \
        $(OBJDIR)/XrdCmsSupervisor.o  \
        $(OBJDIR)/XrdCmsXmiReq.o

BINARY  = $(BINDIR)/cmsd

LIBRARY = $(LIBDIR)/libXrdCms.a

TARGETS = $(BINARY) $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XProtocol% ../XProtocol
vpath XPtypes%   ../XProtocol
vpath YProtocol% ../XProtocol
vpath XrdCms%    ../XrdCms
vpath XrdFrm%    ../XrdFrm
vpath XrdNet%    ../XrdNet
vpath XrdOss%    ../XrdOss
vpath XrdOuc%    ../XrdOuc
vpath XrdSec%    ../XrdSec
vpath XrdSfs%    ../XrdSfs
vpath XrdSys%    ../XrdSys
vpath XrdXrootd% ../XrdXrootd
vpath Xrd%       ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdCms done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(BINARY): $(OBJSRVR) $(OBJBOTH) $(LIBDEPS)
	@echo Creating executable $(BINARY) 
	$(ECHO)$(LD) $(LDOP) $(OBJSRVR) $(OBJBOTH) $(BINLIBS) $(LIBS) $(TYPELIBMT) -o $(BINARY)


$(LIBRARY): $(OBJCLNT) $(OBJBOTH)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJCLNT) $(OBJBOTH)
	$(ECHO)ranlib $(LIBRARY)

$(OBJDIR)/XrdCmsAdmin.o:     XrdCmsAdmin.cc      XrdCmsAdmin.hh \
                             XrdCmsState.hh      XrdCmsConfig.hh \
                             XrdCmsManager.hh    XrdCmsPrepare.hh \
                             XrdCmsProtocol.hh   XrdCmsRRData.hh \
                             XrdCmsTrace.hh \
                             XrdNetSocket.hh     XrdOuca2x.hh \
                             XrdOucName2Name.hh  XrdOucStream.hh \
                             XrdSysError.hh      XrdSysPlatform.hh \
                             XrdSysPthread.hh    YProtocol.hh
	@echo Compiling XrdCmsAdmin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsAdmin.o XrdCmsAdmin.cc

$(OBJDIR)/XrdCmsBaseFS.o:    XrdCmsBaseFS.cc     XrdCmsBaseFS.hh \
                             XrdCmsConfig.hh     XrdCmsPList.hh \
                             XrdCmsPrepare.hh    XrdCmsRRData.hh \
                             XrdCmsTrace.hh      XrdCmsTypes.hh \
                             XrdOss.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysTimer.hh \
                             XPtypes.hh          YProtocol.hh
	@echo Compiling XrdCmsBaseFS.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsBaseFS.o XrdCmsBaseFS.cc

$(OBJDIR)/XrdCmsCache.o:     XrdCmsCache.cc      XrdCmsCache.hh \
                             XrdCmsKey.hh        XrdCmsNash.hh \
                             XrdCmsPList.hh      XrdCmsRRQ.hh \
                             XrdCmsSelect.hh     XrdCmsTypes.hh \
                             XrdSysPthread.hh    XrdSysTimer.hh \
                             XrdJob.hh           XrdScheduler.hh
	@echo Compiling XrdCmsCache.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsCache.o XrdCmsCache.cc

$(OBJDIR)/XrdCmsClientConfig.o: XrdCmsClientConfig.cc   XrdCmsClientConfig.hh \
                                XrdCmsClientMsg.hh      XrdCmsSecurity.hh \
                                XrdCmsTrace.hh \
                                XrdNetDNS.hh            XrdSysHeaders.hh \
                                XrdOuca2x.hh            XrdOucEnv.hh \
                                XrdOucStream.hh         XrdOucTList.hh \
                                XrdOucUtils.hh
	@echo Compiling XrdCmsClientConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsClientConfig.o XrdCmsClientConfig.cc

$(OBJDIR)/XrdCmsClientMsg.o: XrdCmsClientMsg.cc    XrdCmsClientMsg.hh \
                             XrdCmsParser.hh       XrdCmsTrace.hh \
                             XrdOucErrInfo.hh \
                             XrdNetBuffer.hh \
                             XrdSysPthread.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsClientMsg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsClientMsg.o XrdCmsClientMsg.cc

$(OBJDIR)/XrdCmsClientMan.o: XrdCmsClientMan.cc   XrdCmsClientMan.hh \
                             XrdCmsClientMsg.hh   XrdCmsLogin.hh \
                             XrdCmsResp.hh        XrdCmsTrace.hh \
                             XrdNetBuffer.hh \
                             XrdOucErrInfo.hh \
                             XrdSysError.hh       XrdSysPthread.hh \
                             XrdSysTimer.hh \
                             XrdInet.hh           XrdLink.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsClientMan.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsClientMan.o XrdCmsClientMan.cc

$(OBJDIR)/XrdCmsCluster.o:   XrdCmsCluster.cc    XrdCmsCluster.hh \
                             XrdCmsBaseFS.hh \
                             XrdCmsCache.hh      XrdCmsConfig.hh \
                             XrdCmsKey.hh        XrdCmsNode.hh \
                             XrdCmsSelect.hh     XrdCmsState.hh \
                             XrdCmsTrace.hh      XrdCmsTypes.hh \
                             XrdNetDNS.hh        XrdOucPup.hh \
                             XrdOucTList.hh \
                             XrdSysPlatform.hh   XrdSysPthread.hh \
                             XrdSysTimer.hh \
                             XrdJob.hh           XrdLink.hh \
                             XrdScheduler.hh     YProtocol.hh
	@echo Compiling XrdCmsCluster.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsCluster.o XrdCmsCluster.cc

$(OBJDIR)/XrdCmsConfig.o:    XrdCmsConfig.cc     XrdCmsConfig.hh \
                             XrdCmsAdmin.hh      XrdCmsCache.hh \
                             XrdCmsCluster.hh    XrdCmsManager.hh \
                             XrdCmsManTree.hh    XrdCmsMeter.hh \
                             XrdCmsNode.hh       XrdCmsPList.hh \
                             XrdCmsPrepare.hh    XrdCmsPrepArgs.hh \
                             XrdCmsProtocol.hh   XrdCmsRRQ.hh \
                             XrdCmsSecurity.hh \
                             XrdCmsState.hh      XrdCmsSupervisor.hh \
                             XrdCmsTrace.hh      XrdCmsTypes.hh \
                             XrdCmsXmi.hh        XrdCmsXmiReq.hh \
                             XrdNetDNS.hh        XrdNetOpts.hh \
                             XrdNetSecurity.hh   XrdNetSocket.hh \
                             XrdOss.hh \
                             XrdOuca2x.hh        XrdOucEnv.hh \
                             XrdOucExport.hh     XrdOucName2Name.hh \
                             XrdOucPList.hh      XrdOucProg.hh \
                             XrdOucStream.hh     XrdOucTList.hh \
                             XrdOucUtils.hh      XrdSysHeaders.hh \
                             XrdSysError.hh      XrdSysPlatform.hh \
                             XrdSysPlugin.hh     XrdSysPthread.hh \
                             XrdSysTimer.hh \
                             XrdJob.hh           XrdScheduler.hh \
                             ../XrdVersion.hh
	@echo Compiling XrdCmsConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsConfig.o XrdCmsConfig.cc

$(OBJDIR)/XrdCmsFinder.o:    XrdCmsFinder.cc     XrdCmsFinder.hh \
                             XrdCmsClient.hh     XrdCmsClientConfig.hh \
                             XrdCmsClientMan.hh  XrdCmsClientMsg.hh \
                             XrdCmsParser.hh     XrdCmsResp.hh \
                             XrdCmsRRData.hh     XrdCmsTrace.hh \
                             XrdNetDNS.hh        XrdNetOpts.hh \
                             XrdNetSocket.hh \
                             XrdOucEnv.hh        XrdOucErrInfo.hh \
                             XrdOucReqID.hh  \
                             XrdOucStream.hh     XrdOucUtils.hh \
                             XrdSfsInterface.hh \
                             XrdSysError.hh      XrdSysPlatform.hh \
                             XrdSysPthread.hh    XrdSysTimer.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsFinder.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsFinder.o XrdCmsFinder.cc

$(OBJDIR)/XrdCmsJob.o:       XrdCmsJob.cc        XrdCmsJob.hh \
                             XrdCmsProtocol.hh   XrdCmsRRData.hh \
                             XrdCmsTrace.hh      XrdOss.hh \
                             XrdSysHeaders.hh    XrdSysPthread.hh \
                             XrdJob.hh           XrdLink.hh \
                             XrdScheduler.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsJob.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsJob.o XrdCmsJob.cc

$(OBJDIR)/XrdCmsKey.o:       XrdCmsKey.cc        XrdCmsKey.hh \
                             XrdCmsTypes.hh      XrdCmsTrace.hh \
                             XrdOucCRC.hh \
                             XrdSysError.hh
	@echo Compiling XrdCmsKey.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsKey.o XrdCmsKey.cc

$(OBJDIR)/XrdCmsLogin.o:     XrdCmsLogin.cc      XrdCmsLogin.hh \
                             XrdCmsParser.hh     XrdCmsSecurity.hh \
                             XrdCmsTalk.hh       XrdCmsTrace.hh \
                             XrdOucPup.hh \
                             XrdSysPthread.hh    XrdSysError.hh \
                             XrdLink.hh \
                             XPtypes.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsLogin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsLogin.o XrdCmsLogin.cc

$(OBJDIR)/XrdCmsManList.o:   XrdCmsManList.cc    XrdCmsManList.hh \
                             XrdOucPup.hh        XrdNetDNS.hh \
                             XrdSysPlatform.hh   XrdSysPthread.hh
	@echo Compiling XrdCmsManList.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsManList.o XrdCmsManList.cc

$(OBJDIR)/XrdCmsManTree.o:   XrdCmsManTree.cc    XrdCmsManTree.hh \
                             XrdCmsManager.hh    XrdCmsNode.hh \
                             XrdCmsTrace.hh      XrdCmsCluster.hh \
                             XrdSysPthread.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsManTree.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsManTree.o XrdCmsManTree.cc

$(OBJDIR)/XrdCmsManager.o:   XrdCmsManager.cc    XrdCmsManager.hh \
                             XrdCmsConfig.hh     XrdCmsNode.hh \
                             XrdCmsRouting.hh    XrdCmsTrace.hh \
                             XrdCmsManList.hh    XrdCmsTypes.hh \
                             XrdSysPthread.hh    XrdSysTimer.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsManager.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsManager.o XrdCmsManager.cc

$(OBJDIR)/XrdCmsMeter.o:     XrdCmsMeter.cc      XrdCmsMeter.hh \
                             XrdCmsCluster.hh    XrdCmsConfig.hh \
                             XrdCmsNode.hh \
                             XrdCmsState.hh      XrdCmsTrace.hh \
                             XrdOss.hh \
                             XrdOucStream.hh \
                             XrdSysError.hh      XrdSysPlatform.hh \
                             XrdSysPthread.hh
	@echo Compiling XrdCmsMeter.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsMeter.o XrdCmsMeter.cc

$(OBJDIR)/XrdCmsNash.o:      XrdCmsNash.cc       XrdCmsNash.hh \
                             XrdCmsKey.hh
	@echo Compiling XrdCmsNash.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsNash.o XrdCmsNash.cc

$(OBJDIR)/XrdCmsNode.o:      XrdCmsNode.cc       XrdCmsNode.hh \
                             XrdCmsBaseFS.hh \
                             XrdCmsCache.hh      XrdCmsCluster.hh \
                             XrdCmsConfig.hh     XrdCmsKey.hh \
                             XrdCmsManager.hh \
                             XrdCmsManList.hh    XrdCmsManTree.hh \
                             XrdCmsMeter.hh      XrdCmsPList.hh \
                             XrdCmsPrepare.hh    XrdCmsRRData.hh \
                             XrdCmsRRQ.hh        XrdCmsSelect.hh \
                             XrdCmsState.hh      XrdCmsTrace.hh \
                             XrdCmsTypes.hh      XrdCmsXmi.hh \
                             XrdOss.hh           XrdNetDNS.hh \
                             XrdOucName2Name.hh  XrdOucProg.hh \
                             XrdOucPup.hh        XrdOucTokenizer.hh \
                             XrdOucUtils.hh \
                             XrdSysPlatform.hh   XrdSysPthread.hh \
                             XrdJob.hh           XrdLink.hh  \
                             YProtocol.hh
	@echo Compiling XrdCmsNode.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsNode.o XrdCmsNode.cc

$(OBJDIR)/XrdCmsParser.o:    XrdCmsParser.cc     XrdCmsParser.hh \
                             XrdCmsRRData.hh \
                             XrdOucErrInfo.hh    XrdOucPup.hh \
                             XrdSysError.hh      YProtocol.hh
	@echo Compiling XrdCmsParser.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsParser.o XrdCmsParser.cc

$(OBJDIR)/XrdCmsPList.o:     XrdCmsPList.cc      XrdCmsPList.hh \
                             XrdCmsTypes.hh \
                             XrdSysPthread.hh
	@echo Compiling XrdCmsPList.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsPList.o XrdCmsPList.cc

$(OBJDIR)/XrdCmsPrepArgs.o:  XrdCmsPrepArgs.cc   XrdCmsPrepArgs.hh \
                             XrdCmsConfig.hh     XrdCmsNode.hh \
                             XrdCmsPrepare.hh    XrdCmsRRData.hh \
                             XrdCmsTrace.hh \
                             XrdSysPthread.hh \
                             XrdJob.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsPrepArgs.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsPrepArgs.o XrdCmsPrepArgs.cc

$(OBJDIR)/XrdCmsPrepare.o:   XrdCmsPrepare.cc    XrdCmsPrepare.hh \
                             XrdCmsConfig.hh     XrdCmsPrepArgs.hh \
                             XrdCmsTrace.hh      XrdNetMsg.hh \
                             XrdFrmProxy.hh      XrdOss.hh \
                             XrdOucEnv.hh        XrdOucHash.hh \
                             XrdOucMsubs.hh      XrdOucStream.hh \
                             XrdOucTList.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdJob.hh           XrdScheduler.hh
	@echo Compiling XrdCmsPrepare.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsPrepare.o XrdCmsPrepare.cc

$(OBJDIR)/XrdCmsProtocol.o:  XrdCmsProtocol.cc   XrdCmsProtocol.hh \
                             XrdCmsCache.hh      XrdCmsCluster.hh \
                             XrdCmsConfig.hh     XrdCmsJob.hh \
                             XrdCmsLogin.hh      XrdCmsManager.hh \
                             XrdCmsManTree.hh    XrdCmsMeter.hh \
                             XrdCmsParser.hh     XrdCmsRouting.hh \
                             XrdCmsRTable.hh     XrdCmsState.hh \
                             XrdCmsTypes.hh      XrdCmsTrace.hh \
                             XrdOucCRC.hh \
                             XrdOucPup.hh        XrdOucTokenizer.hh \
                             XrdNetDNS.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysHeaders.hh    XrdSysTimer.hh \
                             XrdInet.hh          XrdLink.hh \
                             XrdProtocol.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsProtocol.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsProtocol.o XrdCmsProtocol.cc

$(OBJDIR)/XrdCmsReq.o:       XrdCmsReq.cc        XrdCmsReq.hh \
                             XrdCmsNode.hh       XrdCmsRRQ.hh \
                             XrdCmsRTable.hh     XrdCmsTrace.hh \
                             XrdSysError.hh      XrdSysPlatform.hh \
                             XrdSysPthread.hh \
                             XrdXrootdStat.icc \
                             XProtocol.hh        YProtocol.hh
	@echo Compiling XrdCmsReq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsReq.o XrdCmsReq.cc

$(OBJDIR)/XrdCmsResp.o:    XrdCmsResp.cc      XrdCmsResp.hh \
                           XrdCmsClientMsg.hh XrdCmsParser.hh \
                           XrdCmsTrace.hh \
                           XrdNetBuffer.hh \
                           XrdOucErrInfo.hh   XrdSfsInterface.hh \
                           XrdSysError.hh     XrdSysPthread.hh \
                           YProtocol.hh
	@echo Compiling XrdCmsResp.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsResp.o XrdCmsResp.cc

$(OBJDIR)/XrdCmsRRData.o:    XrdCmsRRData.cc     XrdCmsRRData.hh \
                             XrdSysPlatform.hh   YProtocol.hh
	@echo Compiling XrdCmsRRData.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdCmsRRData.o XrdCmsRRData.cc

$(OBJDIR)/XrdCmsRRQ.o:       XrdCmsRRQ.cc        XrdCmsRRQ.hh \
                             XrdCmsCluster.hh    XrdCmsNode.hh \
                             XrdCmsRTable.hh     XrdCmsTrace.hh \
                             XrdCmsTypes.hh \
                             XrdOucDLlist.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysTimer.hh \
                             XPtypes.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsRRQ.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsRRQ.o XrdCmsRRQ.cc

$(OBJDIR)/XrdCmsRTable.o:    XrdCmsRTable.cc     XrdCmsRTable.hh \
                             XrdCmsNode.hh       XrdCmsTrace.hh  XrdCmsTypes.hh\
                             XrdSysPthread.hh
	@echo Compiling XrdCmsRTable.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsRTable.o XrdCmsRTable.cc

$(OBJDIR)/XrdCmsRouting.o:   XrdCmsRouting.cc    XrdCmsRouting.hh \
                             XrdCmsNode.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsRouting.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsRouting.o XrdCmsRouting.cc

$(OBJDIR)/XrdCmsSecurity.o:  XrdCmsSecurity.cc   XrdCmsSecurity.hh \
                             XrdCmsTalk.hh       XrdCmsTrace.hh \
                             XrdOucEnv.hh        XrdOucErrInfo.hh \
                             XrdOucTList.hh \
                             XrdSecInterface.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdLink.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsSecurity.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsSecurity.o XrdCmsSecurity.cc

$(OBJDIR)/XrdCmsState.o:     XrdCmsState.cc      XrdCmsState.hh \
                             XrdCmsManager.hh \
                             XrdCmsRTable.hh     XrdCmsTalk.hh \
                             XrdCmsTrace.hh      XrdCmsTypes.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdLink.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsState.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsState.o XrdCmsState.cc

$(OBJDIR)/XrdCmsSupervisor.o: XrdCmsSupervisor.cc      XrdCmsSupervisor.hh \
                              XrdInet.hh          XrdLink.hh \
                              XrdCmsConfig.hh     XrdCmsProtocol.hh \
                              XrdCmsTrace.hh \
                              XrdNetSocket.hh \
                              XrdSysError.hh
	@echo Compiling XrdCmsSupervisor.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsSupervisor.o XrdCmsSupervisor.cc

$(OBJDIR)/XrdCmsTalk.o:      XrdCmsTalk.cc       XrdCmsTalk.hh \
                             XrdLink.hh \
                             YProtocol.hh
	@echo Compiling XrdCmsTalk.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsTalk.o XrdCmsTalk.cc

$(OBJDIR)/XrdCmsXmiReq.o:    XrdCmsXmiReq.cc     XrdCmsXmiReq.hh \
                             XrdCmsReq.hh        XrdCmsTrace.hh \
                             XrdCmsXmi.hh \
                             XrdSysPthread.hh
	@echo Compiling XrdCmsXmiReq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCmsXmiReq.o XrdCmsXmiReq.cc
