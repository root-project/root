#              $Id: GNUmakefile,v 1.1 2009/01/14 03:35:14 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdOss -lXrdNet -lXrdOuc -lXrdSys $(TYPELIBSF)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdFrmConfig.cc    \
        XrdFrmPstg.cc      \
        XrdFrmPstgMain.cc  \
        XrdFrmPstgReq.cc   \
        XrdFrmPstgXfr.cc

OBJECTS = \
        $(OBJDIR)/XrdFrmConfig.o    \
        $(OBJDIR)/XrdFrmPstg.o      \
        $(OBJDIR)/XrdFrmPstgMain.o  \
        $(OBJDIR)/XrdFrmPstgReq.o   \
        $(OBJDIR)/XrdFrmPstgXfr.o

OBJPSTG = \
        $(OBJDIR)/XrdXrootdMonitor.o \
        $(OBJDIR)/XrdScheduler.o

OBJDEPS = \
          $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdOss.a \
          $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a

LIBRARY = $(LIBDIR)/libXrdFrm.a

BINARY1 = $(BINDIR)/frm_pstga
BINARY2 = $(BINDIR)/frm_pstgd

TARGETS = $(LIBRARY) $(BINARY1) $(BINARY2)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

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

$(LIBRARY): $(OBJECTS) $(OBJPSTG)
	@echo Creating archive $(LIBRARY)
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		  if [ "x$(SUNCACHE)" != "x" ]; then \
		     $(CC) -xar -o $(LIBRARY) $(OBJDIR)$(SUNCACHE)/*/*.o; \
		  else \
		     $(CC) -xar -o $(LIBRARY) $(OBJECTS); \
		  fi; \
	       else \
	          ar -rc $(LIBRARY) $(OBJECTS); \
	          ranlib $(LIBRARY); \
	       fi

$(BINARY1): $(OBJECTS) $(OBJPSTG) $(OBJDEPS)
	@echo Creating executable $(BINARY1)
	$(ECHO)$(LD) $(LDOP) $(OBJECTS) $(OBJPSTG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINARY1)

$(BINARY2): $(OBJECTS) $(OBJPSTG) $(OBJDEPS)
	@echo Creating executable $(BINARY2)
	$(ECHO)$(LD) $(LDOP) $(OBJECTS) $(OBJPSTG) $(OBJADDS) $(LIBS) $(BINLIBS) $(TYPELIBMT) -o $(BINARY2)

$(OBJDIR)/XrdFrmConfig.o: XrdFrmConfig.hh XrdFrmConfig.cc    XrdFrmTrace.hh \
                          XrdInfo.hh      XrdNetDNS.hh       XrdOss.hh \
                          XrdOuca2x.hh    XrdOucEnv.hh       XrdOucStream.hh \
                          XrdOucUtils.hh  XrdOucName2Name.hh XrdOucMsubs.hh \
                          XrdSysError.hh  XrdSysHeaders.hh   XrdSysLogger.hh \
                          XrdSysTimer.hh  XrdSysPlugin.hh    XrdSysPthread.hh \
                          XrdSysPlatform.hh XrdXrootdMonitor.hh
	@echo Compiling XrdFrmConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdFrmConfig.o XrdFrmConfig.cc

$(OBJDIR)/XrdFrmPstg.o:   XrdFrmPstg.hh    XrdFrmPstg.cc    XrdFrmTrace.hh \
                          XrdFrmConfig.hh  XrdFrmPstgReq.hh XrdFrmPstgXfr.hh \
                          XrdNetMsg.hh     XrdOucStream.hh \
                          XrdSysHeaders.hh XrdSysPlatform.hh XrdSysPthread.hh
	@echo Compiling XrdFrmPstg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstg.o XrdFrmPstg.cc

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
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgMain.o XrdFrmPstgMain.cc

$(OBJDIR)/XrdFrmPstgReq.o: XrdFrmPstgReq.cc      XrdFrmPstgReq.hh \
                           XrdFrmConfig.hh       XrdFrmTrace.hh \
                           XrdSysError.hh        XrdSysPlatform.hh
	@echo Compiling XrdFrmPstgReq.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgReq.o XrdFrmPstgReq.cc

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
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdFrmPstgXfr.o XrdFrmPstgXfr.cc
