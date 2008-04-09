#              $Id: GNUmakefile,v 1.27 2007/08/02 01:39:57 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdOfs.cc  \
        XrdOfsConfig.cc \
        XrdOfsEvr.cc    \
        XrdOfsEvs.cc    \
        XrdOfsHandle.cc

OBJECTS  = \
        $(OBJDIR)/XrdOfs.o \
        $(OBJDIR)/XrdOfsConfig.o  \
        $(OBJDIR)/XrdOfsEvr.o     \
        $(OBJDIR)/XrdOfsEvs.o     \
        $(OBJDIR)/XrdOfsHandle.o  

MORELIBS = -L$(LIBDIR) -lXrdAcc -lXrdOss -lXrdOdc -lXrdNet -lXrdOuc -lXrdSys

LIBDEP   = $(LIBDIR)/libXrdAcc.a $(LIBDIR)/libXrdOss.a \
           $(LIBDIR)/libXrdOdc.a $(LIBDIR)/libXrdOuc.a \
           $(LIBDIR)/libXrdSys.a

LIBRARY  = $(LIBDIR)/libXrdOfs.$(TYPESHLIB)

TARGETS  = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOdc% ../XrdOdc
vpath XrdOfs% ../XrdOfs
vpath XrdOss% ../XrdOss
vpath XrdOuc% ../XrdOuc
vpath XrdAcc% ../XrdAcc
vpath XrdSec% ../XrdSec
vpath XrdSfs% ../XrdSfs
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdOfs done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEP)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(LDSO) $(MORELIBS) $(LIBS)  -o $(LIBRARY)

$(OBJDIR)/XrdOfs.o:  XrdOfs.cc        XrdOfs.hh \
                     XrdOfsConfig.hh  XrdOfsHandle.hh       XrdOfsEvs.hh       \
                     XrdOfsSecurity.hh     XrdOfsTrace.hh   XrdOfsEvr.hh   \
                     XrdOuca2x.hh     XrdOucEnv.hh          XrdSysError.hh     \
                     XrdOucHash.hh    XrdOucHash.icc        XrdOucMsubs.hh \
                     XrdOucLock.hh    XrdSysLogger.hh       XrdSysPlugin.hh \
                     XrdOucPList.hh   XrdSysPthread.hh      XrdOucTrace.hh     \
                     XrdNetDNS.hh     XrdSecEntity.hh       XrdSecInterface.hh \
                     XrdSfsNative.hh  XrdSfsAio.hh     XrdOss.hh          \
                     XrdOucTList.hh   XrdOdcFinder.hh  XrdAccAuthorize.hh      \
                  ../XrdVersion.hh
	@echo Compiling XrdOfs.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfs.o XrdOfs.cc

$(OBJDIR)/XrdOfsConfig.o:  XrdOfsConfig.cc   XrdOfsConfig.hh \
                           XrdOfs.hh         XrdOfsEvs.hh     XrdOfsTrace.hh  \
                           XrdOuca2x.hh      XrdSysError.hh   XrdOucStream.hh \
                           XrdOucTrace.hh    XrdOucUtils.hh   XrdNetDNS.hh   \
                           XrdOucEnv.hh      XrdAccAuthorize.hh \
                           XrdOdcFinder.hh
	@echo Compiling XrdOfsConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsConfig.o XrdOfsConfig.cc

$(OBJDIR)/XrdOfsEvr.o:  XrdOfsEvr.cc      XrdOfsEvr.hh \
                        XrdOfsTrace.hh    XrdOucTrace.hh \
                        XrdOucErrInfo.hh  XrdSysPthread.hh  XrdOucStream.hh \
                        XrdSysTimer.hh    XrdOdcFinder.hh \
                        XrdNetOpts.hh     XrdNetSocket.hh
	@echo Compiling XrdOfsEvr.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsEvr.o XrdOfsEvr.cc

$(OBJDIR)/XrdOfsEvs.o:  XrdOfsEvs.cc      XrdOfsEvs.hh \
                        XrdSysPthread.hh  XrdOucProg.hh \
                        XrdNetOpts.hh     XrdNetSocket.hh
	@echo Compiling XrdOfsEvs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsEvs.o XrdOfsEvs.cc

$(OBJDIR)/XrdOfsHandle.o:  XrdOfsHandle.cc   XrdOfsHandle.hh \
                           XrdOfs.hh         XrdOfsConfig.hh  XrdOfsTrace.hh  \
                           XrdOuca2x.hh      XrdSysError.hh   XrdOucStream.hh \
                           XrdSysPthread.hh  XrdOucTrace.hh   XrdOdcFinder.hh
	@echo Compiling XrdOfsHandle.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsHandle.o XrdOfsHandle.cc
