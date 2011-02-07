#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdOfs.cc  \
        XrdOfsFS.cc  \
        XrdOfsConfig.cc \
        XrdOfsEvr.cc    \
        XrdOfsEvs.cc    \
        XrdOfsHandle.cc \
        XrdOfsPoscq.cc  \
        XrdOfsStats.cc

OBJECTS  = \
        $(OBJDIR)/XrdOfs.o \
        $(OBJDIR)/XrdOfsConfig.o  \
        $(OBJDIR)/XrdOfsEvr.o     \
        $(OBJDIR)/XrdOfsEvs.o     \
        $(OBJDIR)/XrdOfsHandle.o  \
        $(OBJDIR)/XrdOfsPoscq.o   \
        $(OBJDIR)/XrdOfsStats.o

OBJFS    = $(OBJDIR)/XrdOfsFS.o

LIBARCH  = $(LIBDIR)/libXrdOfs.a

TARGETS  = $(LIBARCH)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOfs% ../XrdOfs
vpath XrdOss% ../XrdOss
vpath XrdOuc% ../XrdOuc
vpath XrdAcc% ../XrdAcc
vpath XrdCms% ../XrdCms
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

$(TARGETS): $(OBJECTS) $(OBJFS)
	@echo Creating archive $(LIBARCH)
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJECTS)
	$(ECHO)ranlib $(LIBARCH)

$(OBJDIR)/XrdOfs.o:  XrdOfs.cc             XrdOfs.hh \
                     XrdOfsHandle.hh       XrdOfsEvs.hh     XrdOfsStats.hh \
                     XrdOfsSecurity.hh     XrdOfsTrace.hh   XrdOfsEvr.hh   \
                     XrdOuca2x.hh          XrdOucEnv.hh     XrdSysError.hh     \
                     XrdOucMsubs.hh        XrdCmsClient.hh \
                     XrdOucLock.hh    XrdSysLogger.hh       XrdSysPlugin.hh \
                     XrdOucPList.hh   XrdSysPthread.hh      XrdOucTrace.hh     \
                     XrdNetDNS.hh     XrdSecEntity.hh       XrdSecInterface.hh \
                     XrdSfsNative.hh  XrdSfsAio.hh          XrdSfsInterface.hh \
                     XrdOss.hh        XrdSysHeaders.hh      XrdSysPlatform.hh \
                     XrdOucTList.hh   XrdCmsClient.hh       XrdAccAuthorize.hh
	@echo Compiling XrdOfs.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfs.o XrdOfs.cc

$(OBJDIR)/XrdOfsFS.o:  XrdOfsFS.cc             XrdOfs.hh \
                       XrdSysError.hh          XrdSysPthread.hh \
                    ../XrdVersion.hh
	@echo Compiling XrdOfsFS.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsFS.o XrdOfsFS.cc

$(OBJDIR)/XrdOfsConfig.o:  XrdOfsConfig.cc \
                           XrdOfs.hh         XrdOfsEvs.hh     XrdOfsHandle.hh \
                           XrdOfsPoscq.hh    XrdOfsStats.hh   XrdOfsTrace.hh  \
                           XrdOuca2x.hh      XrdSysError.hh   XrdOucStream.hh \
                           XrdOucTrace.hh    XrdOucUtils.hh   XrdNetDNS.hh   \
                           XrdSysHeaders.hh \
                           XrdOucEnv.hh      XrdAccAuthorize.hh \
                           XrdCmsClient.hh   XrdCmsFinder.hh
	@echo Compiling XrdOfsConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsConfig.o XrdOfsConfig.cc

$(OBJDIR)/XrdOfsEvr.o:  XrdOfsEvr.cc      XrdOfsEvr.hh \
                        XrdOfsStats.hh    XrdOfsTrace.hh   XrdOucTrace.hh \
                        XrdOucErrInfo.hh  XrdOucHash.hh    XrdOucHash.icc \
                        XrdSysPthread.hh  XrdOucStream.hh  XrdOucEnv.hh \
                        XrdSysTimer.hh    XrdCmsClient.hh \
                        XrdNetOpts.hh     XrdNetSocket.hh
	@echo Compiling XrdOfsEvr.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsEvr.o XrdOfsEvr.cc

$(OBJDIR)/XrdOfsEvs.o:  XrdOfsEvs.cc      XrdOfsEvs.hh \
                        XrdSysError.hh    XrdSysPthread.hh \
                        XrdOucEnv.hh      XrdOucProg.hh \
                        XrdNetOpts.hh     XrdNetSocket.hh \
                        XrdSysPlatform.hh
	@echo Compiling XrdOfsEvs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsEvs.o XrdOfsEvs.cc

$(OBJDIR)/XrdOfsHandle.o: XrdOfsHandle.cc   XrdOfsHandle.hh \
                          XrdOfsStats.hh    XrdOss.hh \
                          XrdOucCRC.hh      XrdSysError.hh    XrdSysPlatform.hh\
                          XrdSysPthread.hh  XrdSysTimer.hh
	@echo Compiling XrdOfsHandle.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsHandle.o XrdOfsHandle.cc

$(OBJDIR)/XrdOfsPoscq.o: XrdOfsPoscq.cc    XrdOfsPoscq.hh     XrdOss.hh \
                         XrdSysError.hh    XrdSysPlatform.hh  XrdSysPthread.hh
	@echo Compiling XrdOfsPoscq.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsPoscq.o XrdOfsPoscq.cc

$(OBJDIR)/XrdOfsStats.o: XrdOfsStats.cc    XrdOfsStats.hh  \
                         XrdSysPthread.hh
	@echo Compiling XrdOfsStats.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOfsStats.o XrdOfsStats.cc
