#              $Id: GNUmakefile,v 1.33 2008/09/07 04:34:43 abh Exp $

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
        XrdOssCreate.cc \
        XrdOssLock.cc   \
        XrdOssMio.cc    \
        XrdOssMSS.cc    \
        XrdOssPath.cc \
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
        $(OBJDIR)/XrdOssCreate.o \
        $(OBJDIR)/XrdOssLock.o   \
        $(OBJDIR)/XrdOssMio.o    \
        $(OBJDIR)/XrdOssMSS.o    \
        $(OBJDIR)/XrdOssPath.o   \
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

$(OBJDIR)/XrdOssCache.o: XrdOssApi.hh XrdOssCache.hh XrdOssTrace.hh \
                         XrdOssPath.hh XrdOssSpace.hh \
                         XrdOucDLlist.hh XrdSysHeaders.hh XrdSysError.hh \
                         XrdOssCache.cc
	@echo Compiling XrdOssCache.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssCache.o XrdOssCache.cc

$(OBJDIR)/XrdOssConfig.o: XrdOssApi.hh  XrdOssConfig.hh XrdOssError.hh \
                          XrdOssLock.hh XrdOssTrace.hh XrdSysError.hh \
                          XrdOssMio.hh  XrdOssOpaque.hh XrdOucEnv.hh \
                          XrdOucProg.hh XrdOssConfig.cc XrdOuca2x.hh  \
                          XrdOucMsubs.hh XrdOucName2Name.hh XrdSysPlugin.hh \
                          XrdSysHeaders.hh \
                          XrdOucExport.hh XrdSysPlatform.hh XrdOssCache.hh
	@echo Compiling XrdOssConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssConfig.o XrdOssConfig.cc

$(OBJDIR)/XrdOssCreate.o: XrdOssApi.hh   XrdOssConfig.hh XrdOssError.hh \
                          XrdOssLock.hh  XrdOssTrace.hh  XrdOssOpaque.hh \
                          XrdOssCache.hh XrdOssPath.hh   XrdOucMsubs.hh \
                          XrdOucEnv.hh   XrdSysError.hh  XrdOuca2x.hh  \
                          XrdOucExport.hh XrdSysHeaders.hh \
                          XrdSysPlatform.hh XrdOssCreate.cc
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

$(OBJDIR)/XrdOssPath.o:   XrdOssPath.hh \
                          XrdSysPthread.hh \
                          XrdOssPath.cc
	@echo Compiling XrdOssPath.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssPath.o XrdOssPath.cc

$(OBJDIR)/XrdOssRename.o: XrdOssApi.hh  XrdOssError.hh XrdOssLock.hh \
                          XrdOssPath.hh XrdOssTrace.hh XrdOucExport.hh \
                          XrdOucUtils.hh XrdSysHeaders.hh \
                          XrdOssRename.cc
	@echo Compiling XrdOssRename.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssRename.o XrdOssRename.cc

$(OBJDIR)/XrdOssSpace.o: XrdOssSpace.hh XrdOssSpace.cc \
                         XrdOssCache.hh \
                         XrdOuca2x.hh   XrdOucStream.hh \
                         XrdSysHeaders.hh XrdSysError.hh
	@echo Compiling XrdOssSpace.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssSpace.o XrdOssSpace.cc

$(OBJDIR)/XrdOssStage.o: XrdOssApi.hh   XrdOssStage.hh  XrdOssError.hh \
                         XrdOssLock.hh  XrdOssOpaque.hh XrdOucEnv.hh \
                         XrdOuca2x.hh   XrdOucHash.hh   XrdOucHash.icc \
                         XrdSysPthread.hh XrdOucMsubs.hh XrdOucName2Name.hh \
                         XrdOucDLlist.hh  XrdOucProg.hh  XrdOucReqID.hh \
                         XrdOssStage.cc
	@echo Compiling XrdOssStage.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssStage.o XrdOssStage.cc

$(OBJDIR)/XrdOssStat.o: XrdOssApi.hh       XrdOssCache.hh  XrdOssConfig.hh \
                        XrdOssOpaque.hh    XrdOssPath.hh \
                        XrdOucName2Name.hh XrdOucEnv.hh    XrdSysHeaders.hh \
                        XrdOssStat.cc
	@echo Compiling XrdOssStat.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssStat.o XrdOssStat.cc

$(OBJDIR)/XrdOssUnlink.o: XrdOssApi.hh XrdOssError.hh XrdOssLock.hh \
                          XrdOssOpaque.hh XrdOssPath.hh XrdOssTrace.hh \
                          XrdOucExport.hh XrdSysHeaders.hh \
                          XrdOssUnlink.cc
	@echo Compiling XrdOssUnlink.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdOssUnlink.o XrdOssUnlink.cc
