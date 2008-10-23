#              $Id: GNUmakefile,v 1.19 2008/09/07 04:31:56 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdOdc -l XrdNet -lXrdOuc

LIBDEPS = $(LIBDIR)/libXrdOdc.a $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdOuc.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdOdcConfig.cc \
        XrdOdcFinder.cc \
        XrdOdcManager.cc \
        XrdOdcMsg.cc \
        XrdOdcResp.cc \
        XrdOdcTest.cc
  
OBJECTA = \
        $(OBJDIR)/XrdOdcConfig.o \
        $(OBJDIR)/XrdOdcFinder.o \
        $(OBJDIR)/XrdOdcManager.o \
        $(OBJDIR)/XrdOdcMsg.o \
        $(OBJDIR)/XrdOdcResp.o

OBJECTB = \
        $(OBJDIR)/XrdOdcTest.o

OBJECTS = $(OBJECTA) $(OBJECTB)

LIBRARY = $(LIBDIR)/libXrdOdc.a

# TESTBIN = $(BINDIR)/test_odc

TARGETS = $(LIBRARY) $(TESTBIN)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOdc% ../XrdOdc
vpath XrdOuc% ../XrdOuc
vpath XrdSfs% ../XrdSfs
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdOdc done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJECTA)
	@echo Creating archive $(LIBRARY)
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)if [ "$(TYPE)" = "SunOS" -a "$(CC)" = "CC" ]; then \
		  if [ "x$(SUNCACHE)" != "x" ]; then \
		     $(CC) -xar -o $(LIBRARY) $(OBJDIR)$(SUNCACHE)/*/*.o; \
		  else \
		     $(CC) -xar -o $(LIBRARY) $(OBJECTA); \
		  fi; \
	       else \
	          ar -rc $(LIBRARY) $(OBJECTA); \
	          ranlib $(LIBRARY); \
	       fi

$(TESTBIN): $(OBJECTB) $(LIBDEPS)
	@echo Creating executable $(TESTBIN) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTB) $(BINLIBS) $(LIBS) -o $(TESTBIN)

$(OBJDIR)/XrdOdcConfig.o: XrdOdcConfig.hh XrdOdcMsg.hh     XrdOdcTrace.hh \
                          XrdOuca2x.hh    XrdOucStream.hh  XrdOucTList.hh  \
                          XrdOucEnv.hh    XrdOucUtils.hh   XrdNetWork.hh \
                          XrdSysHeaders.hh \
                          XrdOdcConfig.cc XrdOdcConfDefs.hh
	@echo Compiling XrdOdcConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcConfig.o XrdOdcConfig.cc

$(OBJDIR)/XrdOdcFinder.o: XrdOdcFinder.cc   XrdOdcFinder.hh XrdOdcManager.hh \
                          XrdOdcMsg.hh      XrdOdcTrace.hh    XrdOucEnv.hh \
                          XrdSysPthread.hh  XrdOdcConfDefs.hh XrdSysError.hh \
                          XrdOucErrInfo.hh  XrdSysPlatform.hh XrdSysPthread.hh \
                          XrdOucReqID.hh    XrdOucStream.hh   XrdSysTimer.hh \
                          XrdNetDNS.hh      XrdNetOpts.hh     XrdNetSocket.hh \
                          XrdSfsInterface.hh
	@echo Compiling XrdOdcFinder.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcFinder.o XrdOdcFinder.cc

$(OBJDIR)/XrdOdcManager.o: XrdOdcManager.hh  XrdOdcManager.cc \
                           XrdOdcMsg.hh      XrdOdcResp.hh    XrdOdcTrace.hh \
                           XrdSysError.hh    XrdSysPthread.hh \
                           XrdNetLink.hh     XrdNetWork.hh
	@echo Compiling XrdOdcManager.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcManager.o XrdOdcManager.cc

$(OBJDIR)/XrdOdcMsg.o: XrdOdcMsg.hh     XrdOdcTrace.hh  XrdOucErrInfo.hh \
                       XrdSysPthread.hh \
                       XrdOdcMsg.cc
	@echo Compiling XrdOdcMsg.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcMsg.o XrdOdcMsg.cc

$(OBJDIR)/XrdOdcResp.o: XrdOdcResp.hh     XrdOdcTrace.hh  XrdOucErrInfo.hh \
                       XrdSysPthread.hh   XrdSfsInterface.hh  XrdOdcMsg.hh \
                       XrdOdcResp.cc
	@echo Compiling XrdOdcResp.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcResp.o XrdOdcResp.cc

$(OBJDIR)/XrdOdcTest.o: XrdSysError.hh   XrdOdcFinder.hh  XrdOucErrInfo.hh \
                        XrdSysHeaders.hh XrdSysLogger.hh  XrdOucStream.hh   \
                        XrdOdcTest.cc
	@echo Compiling XrdOdcTest.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOdcTest.o XrdOdcTest.cc
