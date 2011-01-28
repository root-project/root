#              $Id: GNUmakefile,v 1.39 2010/08/17 01:38:42 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdOssAio.cc \
        XrdOssApi.cc \
        XrdOssCache.cc \
        XrdOssConfig.cc \
        XrdOssCopy.cc   \
        XrdOssCreate.cc \
        XrdOssLock.cc   \
        XrdOssMio.cc    \
        XrdOssMSS.cc    \
        XrdOssPath.cc \
        XrdOssReloc.cc  \
        XrdOssRename.cc \
        XrdOssSpace.cc  \
        XrdOssStage.cc  \
        XrdOssStat.cc  \
        XrdOssUnlink.cc

OBJECTS = \
        $(OBJDIR)/XrdOssAio.o \
        $(OBJDIR)/XrdOssApi.o \
        $(OBJDIR)/XrdOssCache.o \
        $(OBJDIR)/XrdOssConfig.o \
        $(OBJDIR)/XrdOssCopy.o   \
        $(OBJDIR)/XrdOssCreate.o \
        $(OBJDIR)/XrdOssLock.o   \
        $(OBJDIR)/XrdOssMio.o    \
        $(OBJDIR)/XrdOssMSS.o    \
        $(OBJDIR)/XrdOssPath.o   \
        $(OBJDIR)/XrdOssReloc.o  \
        $(OBJDIR)/XrdOssRename.o \
        $(OBJDIR)/XrdOssSpace.o  \
        $(OBJDIR)/XrdOssStage.o  \
        $(OBJDIR)/XrdOssStat.o  \
        $(OBJDIR)/XrdOssUnlink.o

LIBRARY = $(LIBDIR)/libXrdOss.a

TARGETS = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdFrm% ../XrdFrm
vpath XrdNet% ../XrdNet
vpath XrdOfs% ../XrdOfs
vpath XrdOuc% ../XrdOuc
vpath XrdSfs% ../XrdSfs
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdOss done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS)
	@echo Creating archive $(LIBRARY)
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJECTS)
	$(ECHO)ranlib $(LIBRARY)

$(OBJDIR)/XrdOssAio.o: XrdOssAio.cc    XrdOssTrace.hh  XrdOssApi.hh \
                       XrdSysError.hh  XrdSysPthread.hh XrdSysPlatform.hh \
                       XrdSfsAio.hh
	@echo Compiling XrdOssAio.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSIGWTI) $(INCLUDE) -o $(OBJDIR)/XrdOssAio.o XrdOssAio.cc

$(OBJDIR)/XrdOssApi.o: XrdOss.hh XrdOssApi.hh XrdOssCache.hh XrdOssConfig.hh \
                       XrdOssError.hh  XrdOssLock.hh     XrdOssTrace.hh \
                       XrdOssMio.hh    XrdOssMioFile.hh  XrdOucExport.hh \
                       XrdSysError.hh  XrdOucPList.hh    XrdSysPthread.hh \
                       XrdOucStream.hh XrdSysPlatform.hh XrdOssApi.cc \
                       XrdOucName2Name.hh XrdSysHeaders.hh XrdSysPlugin.hh \
                       ../XrdVersion.hh
	@echo Compiling XrdOssApi.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssApi.o XrdOssApi.cc

$(OBJDIR)/XrdOssCache.o: XrdOssCache.hh  XrdOssOpaque.hh\
                         XrdOssPath.hh   XrdOssSpace.hh   XrdOssTrace.hh\
                         XrdOucDLlist.hh \
                         XrdSysError.hh  XrdSysHeaders.hh XrdSysPlatform.hh\
                         XrdOssCache.cc
	@echo Compiling XrdOssCache.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssCache.o XrdOssCache.cc

$(OBJDIR)/XrdOssConfig.o: XrdOssApi.hh  XrdOssConfig.hh XrdOssError.hh \
                          XrdOssLock.hh XrdOssTrace.hh XrdSysError.hh \
                          XrdOssMio.hh  XrdOssOpaque.hh XrdOssSpace.hh \
                          XrdOucEnv.hh  XrdOucUtils.hh  XrdFrmProxy.hh \
                          XrdOucProg.hh XrdOssConfig.cc XrdOuca2x.hh  \
                          XrdOucMsubs.hh XrdOucName2Name.hh XrdSysPlugin.hh \
                          XrdSysHeaders.hh \
                          XrdOucExport.hh XrdSysPlatform.hh XrdOssCache.hh
	@echo Compiling XrdOssConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssConfig.o XrdOssConfig.cc

$(OBJDIR)/XrdOssCopy.o: XrdOssCopy.hh XrdOssTrace.hh XrdSysError.hh \
                        XrdOssCopy.cc
	@echo Compiling XrdOssCopy.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssCopy.o XrdOssCopy.cc

$(OBJDIR)/XrdOssCreate.o: XrdOssApi.hh    XrdOssConfig.hh  XrdOssCache.hh \
                          XrdOssCopy.hh   XrdOssError.hh   XrdOssLock.hh \
                          XrdOssOpaque.hh XrdOssPath.hh    XrdOssSpace.hh \
                          XrdOssTrace.hh\
                          XrdOuca2x.hh    XrdOucEnv.hh     XrdOucExport.hh \
                          XrdOucUtils.hh \
                          XrdSysError.hh  XrdSysHeaders.hh XrdSysPlatform.hh \
                          XrdOssCreate.cc
	@echo Compiling XrdOssCreate.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssCreate.o XrdOssCreate.cc

$(OBJDIR)/XrdOssLock.o:   XrdOssApi.hh XrdOssConfig.hh XrdOssError.hh \
                          XrdOssLock.hh XrdSysError.hh XrdSysPlatform.hh \
                          XrdSysHeaders.hh XrdOssLock.cc
	@echo Compiling XrdOssLock.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssLock.o XrdOssLock.cc

$(OBJDIR)/XrdOssMio.o:    XrdOssMio.cc   XrdOssMio.hh    XrdOssMioFile.hh \
                          XrdOssTrace.hh XrdSysError.hh  XrdSysPthread.hh \
                          XrdOucHash.hh  XrdOucHash.icc
	@echo Compiling XrdOssMio.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssMio.o XrdOssMio.cc

$(OBJDIR)/XrdOssMSS.o:    XrdOssApi.hh   XrdOssConfig.hh XrdOssError.hh \
                          XrdOssTrace.hh XrdSysError.hh  XrdOucProg.hh \
                          XrdOucStream.hh XrdSysPlatform.hh  XrdSysLogger.hh \
                          XrdNetOpts.hh   XrdNetSocket.hh  XrdSysHeaders.hh \
                          XrdOssMSS.cc
	@echo Compiling XrdOssMSS.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssMSS.o XrdOssMSS.cc

$(OBJDIR)/XrdOssPath.o:   XrdOssPath.hh   XrdOssSpace.hh \
                          XrdSysPthread.hh \
                          XrdOssPath.cc
	@echo Compiling XrdOssPath.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssPath.o XrdOssPath.cc

$(OBJDIR)/XrdOssReloc.o: XrdOssReloc.cc \
                         XrdOssApi.hh     XrdOssCache.hh    XrdOssConfig.hh \
                         XrdOssCopy.hh    XrdOssError.hh    XrdOssPath.hh \
                         XrdOssSpace.hh   XrdOssTrace.hh   XrdOucUtils.hh \
                         XrdSysError.hh   XrdSysHeaders.hh  XrdSysPlatform.hh
	@echo Compiling XrdOssReloc.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssReloc.o XrdOssReloc.cc

$(OBJDIR)/XrdOssRename.o: XrdOssApi.hh  XrdOssError.hh XrdOssLock.hh \
                          XrdOssPath.hh XrdOssCache.hh XrdOssTrace.hh \
                          XrdOucExport.hh \
                          XrdOucUtils.hh XrdSysHeaders.hh \
                          XrdOssRename.cc
	@echo Compiling XrdOssRename.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssRename.o XrdOssRename.cc

$(OBJDIR)/XrdOssSpace.o: XrdOssSpace.hh XrdOssSpace.cc \
                         XrdOssCache.hh \
                         XrdOuca2x.hh     XrdOucEnv.hh   XrdOucStream.hh \
                         XrdOucUtils.hh \
                         XrdSysHeaders.hh XrdSysError.hh XrdSysPlatform.hh
	@echo Compiling XrdOssSpace.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssSpace.o XrdOssSpace.cc

$(OBJDIR)/XrdOssStage.o: XrdOssApi.hh   XrdOssStage.hh  XrdOssError.hh \
                         XrdOssLock.hh  XrdOssOpaque.hh XrdOucEnv.hh \
                         XrdOuca2x.hh   XrdOucHash.hh   XrdOucHash.icc \
                         XrdSysPthread.hh XrdOucMsubs.hh XrdOucName2Name.hh \
                         XrdOucDLlist.hh  XrdOucProg.hh  XrdOucReqID.hh \
                         XrdSysPlatform.hh XrdFrmProxy.hh XrdOssStage.cc
	@echo Compiling XrdOssStage.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssStage.o XrdOssStage.cc

$(OBJDIR)/XrdOssStat.o: XrdOssApi.hh       XrdOssCache.hh  XrdOssConfig.hh \
                        XrdOssOpaque.hh    XrdOssPath.hh   XrdOssSpace.hh \
                        XrdOucName2Name.hh XrdOucEnv.hh    XrdSysHeaders.hh \
                        XrdOssStat.cc
	@echo Compiling XrdOssStat.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssStat.o XrdOssStat.cc

$(OBJDIR)/XrdOssUnlink.o: XrdOssApi.hh XrdOssError.hh XrdOssCache.hh XrdOssLock.hh \
                          XrdOssOpaque.hh XrdOssPath.hh XrdOssTrace.hh \
                          XrdOssConfig.hh \
                          XrdOucExport.hh XrdSysHeaders.hh \
                          XrdOssUnlink.cc
	@echo Compiling XrdOssUnlink.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssUnlink.o XrdOssUnlink.cc
