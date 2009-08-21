#              $Id: GNUmakefile,v 1.4 2009/06/11 15:31:49 ganis Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdFrm -lXrdCms -lXrdOss -lXrdNet -lXrdOuc -lXrdSys $(TYPELIBSF)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdFrmAdmin.cc     \
        XrdFrmAdminAudit.cc\
        XrdFrmAdminFiles.cc \
        XrdFrmAdminFind.cc \
        XrdFrmAdminMain.cc \
        XrdFrmAdminQuery.cc \
        XrdFrmAdminUnlink.cc\
        XrdFrmConfig.cc    \
        XrdFrmFiles.cc     \
        XrdFrmPstg.cc      \
        XrdFrmPstgMain.cc  \
        XrdFrmPstgReq.cc   \
        XrdFrmPstgXfr.cc   \
        XrdFrmUtils.cc

OBJLIB  = \
        $(OBJDIR)/XrdFrmConfig.o    \
        $(OBJDIR)/XrdFrmFiles.o     \
        $(OBJDIR)/XrdFrmUtils.o

OBJADMIN= \
        $(OBJDIR)/XrdFrmAdminMain.o \
        $(OBJDIR)/XrdFrmAdmin.o     \
        $(OBJDIR)/XrdFrmAdminAudit.o\
        $(OBJDIR)/XrdFrmAdminFiles.o\
        $(OBJDIR)/XrdFrmAdminFind.o \
        $(OBJDIR)/XrdFrmAdminQuery.o\
        $(OBJDIR)/XrdFrmAdminUnlink.o

OBJPSTG = \
        $(OBJDIR)/XrdFrmPstgMain.o  \
        $(OBJDIR)/XrdFrmPstg.o      \
        $(OBJDIR)/XrdFrmPstgReq.o   \
        $(OBJDIR)/XrdFrmPstgXfr.o

OBJECTS = $(OBJLIB) $(OBJADMIN) $(OBJPSTG)

OBJADDS = \
        $(OBJDIR)/XrdXrootdMonitor.o \
        $(OBJDIR)/XrdScheduler.o

OBJDEPS = \
          $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdOss.a \
          $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a

LIBRARY = $(LIBDIR)/libXrdFrm.a

BINADMIN= $(BINDIR)/frm_admin
BINPSTGA= $(BINDIR)/frm_pstga
BINPSTGD= $(BINDIR)/frm_pstgd

TARGETS = $(LIBRARY) $(BINADMIN) $(BINPSTGA) $(BINPSTGD)

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

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdFrm done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJLIB) $(OBJPSTG)
	@echo Creating archive $(LIBRARY)
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		  if [ "x$(SUNCACHE)" != "x" ]; then \
		     $(CC) -xar -o $(LIBRARY) $(OBJDIR)$(SUNCACHE)/*/*.o; \
		  else \
		     $(CC) -xar -o $(LIBRARY) $(OBJLIB); \
		  fi; \
	       else \
	          ar -rc $(LIBRARY) $(OBJLIB); \
	          ranlib $(LIBRARY); \
	       fi

$(BINADMIN): $(OBJLIB) $(OBJADMIN)           $(OBJDEPS)
	@echo Creating executable $(BINADMIN)
	$(ECHO)$(LD) $(LDOP) $(OBJADMIN) $(OBJADDS) $(LIBREADLINE) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINADMIN)

$(BINPSTGA): $(OBJLIB) $(OBJPSTG) $(OBJADDS) $(OBJDEPS)
	@echo Creating executable $(BINPSTGA)
	$(ECHO)$(LD) $(LDOP) $(OBJPSTG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINPSTGA)

$(BINPSTGD): $(OBJLIB) $(OBJPSTG) $(OBJADDS) $(OBJDEPS)
	@echo Creating executable $(BINPSTGD)
	$(ECHO)$(LD) $(LDOP) $(OBJPSTG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINPSTGD)

$(OBJDIR)/XrdFrmAdmin.o:  XrdFrmAdmin.hh    XrdFrmAdmin.cc  XrdFrmTrace.hh \
                          XrdFrmConfig.hh   XrdFrmUtils.hh  \
                          XrdOss.hh         XrdOucExport.hh \
                          XrdOuca2x.hh      XrdOucArgs.hh   \
                          XrdOucTList.hh    XrdOucTokenizer.hh
	@echo Compiling XrdFrmAdmin.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdmin.o XrdFrmAdmin.cc

$(OBJDIR)/XrdFrmAdminAudit.o:  XrdFrmAdmin.hh    XrdFrmAdminAudit.cc \
                               XrdFrmConfig.hh   XrdFrmFiles.hh  \
                               XrdFrmTrace.hh    XrdFrmUtils.hh \
                               XrdOssPath.hh     XrdOssSpace.hh \
                               XrdOucNSWalk.hh   XrdOucTList.hh \
                               XrdSysTimer.hh
	@echo Compiling XrdFrmAdminAudit.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminAudit.o XrdFrmAdminAudit.cc

$(OBJDIR)/XrdFrmAdminFiles.o: XrdFrmAdmin.hh    XrdFrmAdminFiles.cc \
                              XrdFrmConfig.hh   XrdFrmFiles.hh  \
                              XrdFrmTrace.hh    XrdOssPath.hh
	@echo Compiling XrdFrmAdminFiles.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminFiles.o XrdFrmAdminFiles.cc

$(OBJDIR)/XrdFrmAdminFind.o:  XrdFrmAdmin.hh    XrdFrmAdminFind.cc \
                              XrdFrmConfig.hh   XrdFrmFiles.hh  \
                              XrdFrmTrace.hh    XrdOssPath.hh \
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
                               XrdFrmTrace.hh    \
                               XrdOssPath.hh     XrdOssSpace.hh \
                               XrdOucArgs.hh     XrdOucNSWalk.hh \
                               XrdOucTList.hh
	@echo Compiling XrdFrmAdminQuery.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminQuery.o XrdFrmAdminQuery.cc

$(OBJDIR)/XrdFrmAdminUnlink.o: XrdFrmAdmin.hh    XrdFrmAdminUnlink.cc \
                               XrdFrmConfig.hh   XrdFrmTrace.hh \
                               XrdFrmUtils.hh  \
                               XrdOss.hh         XrdOssPath.hh \
                               XrdOucNSWalk.hh   XrdCmsNotify.hh
	@echo Compiling XrdFrmAdminUnlink.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmAdminUnlink.o XrdFrmAdminUnlink.cc

$(OBJDIR)/XrdFrmConfig.o: XrdFrmConfig.hh XrdFrmConfig.cc    XrdFrmTrace.hh \
                          XrdCmsNotify.hh \
                          XrdInfo.hh      XrdNetDNS.hh       XrdOss.hh \
                          XrdOuca2x.hh    XrdOucEnv.hh       XrdOucStream.hh \
                          XrdOucUtils.hh  XrdOucName2Name.hh XrdOucMsubs.hh \
                          XrdOucProg.hh   XrdOucTList.hh     XrdOssSpace.hh \
                          XrdSysError.hh  XrdSysHeaders.hh   XrdSysLogger.hh \
                          XrdSysTimer.hh  XrdSysPlugin.hh    XrdSysPthread.hh \
                          XrdSysPlatform.hh XrdXrootdMonitor.hh
	@echo Compiling XrdFrmConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmConfig.o XrdFrmConfig.cc

$(OBJDIR)/XrdFrmFiles.o:  XrdFrmFiles.hh    XrdFrmFiles.cc  XrdFrmTrace.hh \
                          XrdFrmConfig.hh   XrdFrmUtils.hh  XrdOssPath.hh \
                          XrdOucHash.hh     XrdOucNSWalk.hh
	@echo Compiling XrdFrmFiles.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmFiles.o XrdFrmFiles.cc

$(OBJDIR)/XrdFrmPstg.o:   XrdFrmPstg.hh    XrdFrmPstg.cc    XrdFrmTrace.hh \
                          XrdFrmConfig.hh  XrdFrmPstgReq.hh XrdFrmPstgXfr.hh \
                          XrdNetMsg.hh     XrdOucStream.hh \
                          XrdSysHeaders.hh XrdSysPlatform.hh XrdSysPthread.hh
	@echo Compiling XrdFrmPstg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstg.o XrdFrmPstg.cc

$(OBJDIR)/XrdFrmPstgMain.o: XrdFrmPstgMain.cc \
                            XrdFrmConfig.hh      XrdFrmPstg.hh \
                            XrdFrmTrace.hh \
                            XrdFrmPstgReq.hh     XrdFrmPstgXfr.hh \
                            XrdNetOpts.hh        XrdNetSocket.hh \
                            XrdOucUtils.cc \
                            XrdSysError.hh       XrdSysHeaders.hh \
                            XrdSysLogger.hh      XrdSysPthread.hh \
                            XrdSysTimer.hh
	@echo Compiling XrdFrmPstgMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgMain.o XrdFrmPstgMain.cc

$(OBJDIR)/XrdFrmPstgReq.o: XrdFrmPstgReq.cc      XrdFrmPstgReq.hh \
                           XrdFrmConfig.hh       XrdFrmTrace.hh \
                           XrdSysError.hh        XrdSysPlatform.hh
	@echo Compiling XrdFrmPstgReq.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgReq.o XrdFrmPstgReq.cc

$(OBJDIR)/XrdFrmPstgXfr.o: XrdFrmPstgXfr.cc      XrdFrmPstgXfr.hh \
                           XrdFrmConfig.hh       XrdFrmPstgReq.hh \
                           XrdFrmTrace.hh \
                           XrdNetMsg.hh          XrdOss.hh \
                           XrdOucEnv.hh          XrdOucHash.hh \
                           XrdOucMsubs.hh        XrdOucProg.hh \
                           XrdOucTList.hh \
                           XrdSysError.hh        XrdSysPlatform.hh \
                           XrdSysPthread.hh      XrdSysTimer.hh \
                           XrdXrootdMonitor.hh
	@echo Compiling XrdFrmPstgXfr.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgXfr.o XrdFrmPstgXfr.cc

$(OBJDIR)/XrdFrmUtils.o:  XrdFrmUtils.hh    XrdFrmUtils.cc  XrdFrmTrace.hh \
                          XrdSysError.hh
	@echo Compiling XrdFrmUtils.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdFrmUtils.o XrdFrmUtils.cc
