#              $Id$

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

LIBLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys -lpthread

ifneq ($(CFHASXRDCRYPTO),)
BINLIBS = $(LIBDIR)/libXrdClient.a \
          $(LIBDIR)/libXrdCrypto.a  $(LIBDIR)/libXrdSut.a \
          $(LIBDIR)/libXrdNet.a     $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdSys.a
else
BINLIBS = $(LIBDIR)/libXrdClient.a \
          $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdOuc.a 
          $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdSys.a
endif

LIBDEPS = $(BINLIBS)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
	XrdClientAbs.cc	                \
	XrdClient.cc		        \
	XrdClientSock.cc	        \
	XrdClientConn.cc		\
	XrdClientConnMgr.cc		\
	XrdClientDebug.cc		\
	XrdClientInputBuffer.cc	        \
	XrdClientLogConnection.cc	\
	XrdClientMessage.cc		\
	XrdClientPhyConnection.cc	\
	XrdClientReadCache.cc		\
	XrdClientPrep.cc		\
	XrdClientUrlInfo.cc		\
	XrdClientUrlSet.cc		\
	XrdClientThread.cc              \
	XrdClientProtocol.cc            \
	XrdClientAdmin.cc		\
	XrdCpMthrQueue.cc		\
	XrdCpWorkLst.cc                 \
        XrdClientAdmin_c.cc             \
        XrdClientAdmin_c_wrap.c		\
	XrdClientSid.cc                 \
	XrdClientPSock.cc               \
	XrdClientMStream.cc             \
	XrdClientReadV.cc               \
	XrdcpXtremeRead.cc              \
	XrdClientReadAhead.cc

  
OBJECTS = \
	$(OBJDIR)/XrdClientAbs.o                \
	$(OBJDIR)/XrdClient.o                   \
	$(OBJDIR)/XrdClientSock.o               \
	$(OBJDIR)/XrdClientConn.o		\
	$(OBJDIR)/XrdClientConnMgr.o		\
	$(OBJDIR)/XrdClientDebug.o		\
	$(OBJDIR)/XrdClientInputBuffer.o	\
	$(OBJDIR)/XrdClientLogConnection.o	\
	$(OBJDIR)/XrdClientMessage.o		\
	$(OBJDIR)/XrdClientPhyConnection.o	\
	$(OBJDIR)/XrdClientReadCache.o          \
	$(OBJDIR)/XrdClientThread.o             \
	$(OBJDIR)/XrdClientUrlInfo.o		\
	$(OBJDIR)/XrdClientUrlSet.o		\
	$(OBJDIR)/XrdClientProtocol.o           \
	$(OBJDIR)/XrdClientAdmin.o              \
	$(OBJDIR)/XrdClientEnv.o		\
	$(OBJDIR)/XrdCpMthrQueue.o		\
	$(OBJDIR)/XrdCpWorkLst.o		\
	$(OBJDIR)/XrdClientSid.o                \
	$(OBJDIR)/XrdClientPSock.o              \
	$(OBJDIR)/XrdClientMStream.o            \
	$(OBJDIR)/XrdClientReadV.o              \
	$(OBJDIR)/XrdClientReadAhead.o

OBJECTB = $(OBJDIR)/TestXrdClient.o $(OBJDIR)/XrdClientAdmin_c.o

OBJECTC = \
	$(OBJDIR)/Xrdcp.o                     \
	$(OBJDIR)/XrdcpXtremeRead.o

OBJECTD = $(OBJDIR)/TestXrdClient_read.o
OBJECT_JNI = $(OBJDIR)/XrdClientAdminJNI.o

OBJECT_PREPTOOL = $(OBJDIR)/XrdClientPrep.o

OBJECT_STAGETOOL = $(OBJDIR)/XrdStageTool.o

OBJECT_CLI = $(OBJDIR)/XrdCommandLine.o
OBJECTPERLMOD = $(OBJDIR)/XrdClientAdmin_c.o            \
                $(OBJDIR)/XrdClientAdmin_c_wrap.o

LIBRARY = $(LIBDIR)/libXrdClient.a

LIBDEP   = $(LIBDIR)/libXrdNet.a \
           $(LIBDIR)/libXrdOuc.a \
           $(LIBDIR)/libXrdSys.a

LIBSHARED  = $(LIBDIR)/libXrdClient.$(TYPESHLIB)

LIBRARY_JNI =
CLASS_JNI =
ifneq ($(JAVAINC),)
LIBRARY_JNI = $(LIBDIR)/libXrdClientAdminJNI.$(TYPESHLIB)
CLASS_JNI = XrdClientAdminJNI
CLASS_JNI_TARGET = $(LIBDIR)/XrdClientAdminJNI.class
endif

TESTBIN = $(BINDIR)/TestXrdClient $(BINDIR)/TestXrdClient_read
XRDCP = $(BINDIR)/xrdcp
XRDCLI = $(BINDIR)/xrd
XRDADMIN = $(BINDIR)/xrdadmin
XRDPREPTOOL = $(BINDIR)/xprep
XRDSTAGETOOL = $(BINDIR)/xrdstagetool

PERLMOD =
ifneq ($(PERLINC),)
PERLMOD = $(LIBDIR)/XrdClientAdmin.so
endif

TARGETS = $(LIBRARY) $(LIBSHARED) $(LIBAUX) \
	  $(TESTBIN) $(XRDCP) $(XRDCLI) $(XRDADMIN) \
	  $(LIBRARY_JNI) $(CLASS_JNI_TARGET) $(XRDPREPTOOL) $(XRDSTAGETOOL)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdOuc% ../XrdOuc

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS) perlmod
	@echo Make XrdClient done.

perlmod: $(PERLMOD)
	$(ECHO)cp *.pm $(LIBDIR)
	@echo Make perlmodule done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJECTS)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJECTS)
	$(ECHO)ranlib $(LIBRARY)

$(LIBSHARED): $(OBJECTS) $(LIBDEP)
	@echo Creating shared library $(LIBSHARED) 
	$(ECHO)rm -f $(LIBSHARED)
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(LDSO) $(LIBLIBS) $(LIBS) -o $(LIBSHARED)
ifneq ($(subst $(MACOSX_MINOR),,1234),1234)
	$(ECHO)if [ "x$(TYPE)" = "xDarwin" ]; then \
	          $(CC) $(CFLAGS) $(OBJECTS) $(TYPELDDY) $(LIBLIBS) $(LIBS) -o $(LIBDIR)/libXrdClient.$(TYPEDYLIB); \
	fi
endif

$(LIBRARY_JNI): $(OBJECT_JNI)
	@echo Creating shared library $(LIBRARY_JNI) 
	$(ECHO)rm -f $(LIBRARY_JNI);
	$(ECHO)$(CC) $(CFLAGS) $(OBJECT_JNI) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBRARY_JNI)

$(CLASS_JNI_TARGET): $(CLASS_JNI).java
	@echo Creating JNI class file $(CLASS_JNI_TARGET)
	$(ECHO)$(JAVACBIN) -d $(LIBDIR) $(CLASS_JNI).java

$(BINDIR)/TestXrdClient_read: $(OBJECTS) $(OBJECTD) $(LIBDEPS)
	@echo Creating executable $(BINDIR)/TestXrdClient_read 
	$(ECHO)$(LD) $(LDOP) $(OBJECTD) $(BINLIBS) $(LIBS) -o $(BINDIR)/TestXrdClient_read

$(BINDIR)/TestXrdClient: $(OBJECTS) $(OBJECTB) $(LIBDEPS)
	@echo Creating executable $(BINDIR)/TestXrdClient 
	$(ECHO)$(LD) $(LDOP) $(OBJECTB) $(BINLIBS) $(LIBS) -o $(BINDIR)/TestXrdClient

$(XRDCP): $(OBJECTS) $(OBJECTC) $(LIBDEPS)
	@echo Creating executable $(XRDCP) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTC) $(BINLIBS) $(LIBS) $(LIBZ) -o $(XRDCP)

$(XRDPREPTOOL): $(OBJECTS) $(OBJECT_PREPTOOL) $(LIBDEPS)
	@echo Creating executable $(XRDPREPTOOL)
	$(ECHO)$(LD) $(LDOP) $(OBJECT_PREPTOOL) $(BINLIBS) $(LIBS) -o $(XRDPREPTOOL)

$(XRDSTAGETOOL): $(OBJECTS) $(OBJECT_STAGETOOL) $(LIBDEPS)
	@echo Creating executable $(XRDSTAGETOOL) 
	$(ECHO)$(LD) $(LDOP) $(OBJECT_STAGETOOL) $(BINLIBS) $(LIBS) -o $(XRDSTAGETOOL)

$(XRDCLI): $(OBJECTS) $(OBJECT_CLI) $(LIBDEPS)
	@echo Creating executable $(XRDCLI) 
	$(ECHO)$(LD) $(LDOP) $(OBJECT_CLI) $(BINLIBS) $(LIBS) $(LIBREADLINE) -o $(XRDCLI)

$(XRDADMIN): $(PERLMOD)
	$(ECHO)if [ "$(PERLBIN)" != "" ]; then \
		echo Creating executable $(XRDADMIN); \
		sed -e "s|#!/usr/bin/perl|#!$(PERLBIN)|" < xrdadmin > $(XRDADMIN); \
		chmod a+x $(XRDADMIN); \
	fi;

$(PERLMOD): $(OBJECTS) $(OBJECTC) $(OBJECTPERLMOD) $(LIBDEPS)
	@echo Creating perl module library $(PERLMOD) 
	$(ECHO)$(CC) $(OBJECTC) $(OBJECTPERLMOD) $(BINLIBS) $(LDSO) $(PERLCCFLAGS) $(LIBS) -o $(PERLMOD)

$(OBJDIR)/XrdClientAdmin_c_wrap.o:        XrdClientAdmin_c_wrap.c
	@echo Compiling XrdClientAdmin_c_wrap.c
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) $(PERLINC) $(PERLCCFLAGS) -o $(OBJDIR)/XrdClientAdmin_c_wrap.o XrdClientAdmin_c_wrap.c

$(OBJDIR)/XrdClientAdmin_c.o: XrdClientAdmin_c.cc ../XrdClient/XrdClientAdmin.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdSys/XrdSysPlatform.hh
	@echo Compiling XrdClientAdmin_c.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) $(PERLCCFLAGS) -o $(OBJDIR)/XrdClientAdmin_c.o XrdClientAdmin_c.cc

$(OBJDIR)/XrdClientThread.o: XrdClientThread.cc ../XrdClient/XrdClientThread.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh
	@echo Compiling XrdClientThread.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientThread.o XrdClientThread.cc

$(OBJDIR)/XrdClientAbs.o: XrdClientAbs.cc ../XrdClient/XrdClientAbs.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientReadCache.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdOuc/XrdOucHash.hh \
  ../XrdOuc/XrdOucHash.icc ../XrdClient/XrdClientVector.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientAbs.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdClientAbs.o XrdClientAbs.cc

$(OBJDIR)/XrdClient.o: XrdClient.cc ../XrdClient/XrdClient.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientUrlSet.hh \
  ../XrdClient/XrdClientConnMgr.hh ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientSid.hh ../XrdOuc/XrdOucRash.hh \
  ../XrdOuc/XrdOucRash.icc ../XrdSys/XrdSysPlatform.hh \
  ../XrdClient/XrdClientReadAhead.hh
	@echo Compiling XrdClient.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClient.o XrdClient.cc


$(OBJDIR)/XrdClientAdmin.o: XrdClientAdmin.cc ../XrdClient/XrdClientAdmin.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientUrlSet.hh \
  ../XrdClient/XrdClientConnMgr.hh ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientThread.hh
	@echo Compiling XrdClientAdmin.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdClientAdmin.o XrdClientAdmin.cc

$(OBJDIR)/XrdClientAdminJNI.o: XrdClientAdminJNI.cc ../XrdClient/XrdClientAdmin.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientUrlSet.hh \
  ../XrdClient/XrdClientConnMgr.hh ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientThread.hh
	@echo Compiling XrdClientAdminJNI.cc Java include path=$(JAVAINC)
	$(ECHO)$(CC) -g -c -I $(JAVAINC) -I $(JAVAINC)/linux  $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientAdminJNI.o XrdClientAdminJNI.cc

$(OBJDIR)/XrdClientSock.o: XrdClientSock.cc ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh  \
  ../XrdSys/XrdSysLogger.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdNet/XrdNetSocket.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc
	@echo Compiling XrdClientSock.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientSock.o XrdClientSock.cc

$(OBJDIR)/XrdClientPSock.o: XrdClientPSock.cc ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdNet/XrdNetSocket.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc
	@echo Compiling XrdClientPSock.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientPSock.o XrdClientPSock.cc

$(OBJDIR)/XrdClientConn.o: XrdClientConn.cc XrdClientConn.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientConst.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientConnMgr.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdSec/XrdSecInterface.hh ../XrdSec/XrdSecEntity.hh \
  ../XrdNet/XrdNetDNS.hh  \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientSid.hh \
  ../XrdOuc/XrdOucRash.hh ../XrdOuc/XrdOucRash.icc \
  ../XrdSys/XrdSysPlatform.hh
	@echo Compiling XrdClientConn.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdClientConn.o XrdClientConn.cc



$(OBJDIR)/XrdClientConnMgr.o: XrdClientConnMgr.cc ../XrdClient/XrdClientConnMgr.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientConnMgr.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientConnMgr.o XrdClientConnMgr.cc

$(OBJDIR)/XrdClientDebug.o: XrdClientDebug.cc ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientConst.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientDebug.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientDebug.o XrdClientDebug.cc

$(OBJDIR)/XrdClientInputBuffer.o: XrdClientInputBuffer.cc \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdOuc/XrdOucHash.hh \
  ../XrdOuc/XrdOucHash.icc ../XrdClient/XrdClientVector.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientInputBuffer.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientInputBuffer.o XrdClientInputBuffer.cc

$(OBJDIR)/XrdClientLogConnection.o: XrdClientLogConnection.cc \
  ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientPhyConnection.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientSid.hh
	@echo Compiling XrdClientLogConnection.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientLogConnection.o XrdClientLogConnection.cc

$(OBJDIR)/XrdClientMessage.o: XrdClientMessage.cc ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientPhyConnection.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdClient/XrdClientVector.hh \
  ../XrdClient/XrdClientThread.hh
	@echo Compiling XrdClientMessage.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientMessage.o XrdClientMessage.cc

$(OBJDIR)/XrdClientPhyConnection.o: XrdClientPhyConnection.cc \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdOuc/XrdOucHash.hh \
  ../XrdOuc/XrdOucHash.icc ../XrdClient/XrdClientVector.hh \
  ../XrdClient/XrdClientThread.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdSys/XrdSysLogger.hh \
  ../XrdClient/XrdClientSid.hh ../XrdOuc/XrdOucRash.hh \
  ../XrdOuc/XrdOucRash.icc ../XrdSys/XrdSysPlatform.hh
	@echo Compiling XrdClientPhyConnection.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientPhyConnection.o XrdClientPhyConnection.cc

$(OBJDIR)/XrdClientReadCache.o: XrdClientReadCache.cc \
  ../XrdClient/XrdClientReadCache.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientReadCache.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientReadCache.o XrdClientReadCache.cc

$(OBJDIR)/XrdClientUrlInfo.o: XrdClientUrlInfo.cc ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdNet/XrdNetDNS.hh
	@echo Compiling XrdClientUrlInfo.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientUrlInfo.o XrdClientUrlInfo.cc

$(OBJDIR)/XrdClientUrlSet.o: XrdClientUrlSet.cc ../XrdClient/XrdClientUrlSet.hh \
  ../XrdClient/XrdClientUrlInfo.hh  \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientVector.hh \
  ../XrdClient/XrdClientDebug.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdClientUrlSet.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientUrlSet.o XrdClientUrlSet.cc

$(OBJDIR)/XrdClientProtocol.o: XrdClientProtocol.cc ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdSys/XrdSysPlatform.hh
	@echo Compiling XrdClientProtocol.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientProtocol.o XrdClientProtocol.cc

$(OBJDIR)/TestXrdClient.o: TestXrdClient.cc ../XrdClient/XrdClient.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientAdmin_c.hh
	@echo Compiling TestXrdClient.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/TestXrdClient.o TestXrdClient.cc


$(OBJDIR)/TestXrdClient_read.o: TestXrdClient_read.cc ../XrdClient/XrdClient.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientAdmin_c.hh
	@echo Compiling TestXrdClient_read.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/TestXrdClient_read.o TestXrdClient_read.cc


$(OBJDIR)/XrdCpMthrQueue.o: XrdCpMthrQueue.cc ../XrdClient/XrdCpMthrQueue.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClientVector.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysLogger.hh
	@echo Compiling XrdMthrQueue.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCpMthrQueue.o XrdCpMthrQueue.cc

$(OBJDIR)/Xrdcp.o: Xrdcp.cc ../XrdClient/XrdClientUrlInfo.hh \
  ../XrdOuc/XrdOucString.hh ../XrdSys/XrdSysHeaders.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClient.hh ../XrdClient/XrdClientAbs.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientConn.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientConnMgr.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientVector.hh \
  ../XrdOuc/XrdOucRash.hh ../XrdOuc/XrdOucRash.icc \
  ../XrdSys/XrdSysPlatform.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientReadCache.hh ../XrdVersion.hh \
  ../XrdClient/XrdCpMthrQueue.hh ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdCpWorkLst.hh \
  ../XrdClient/XrdClientAdmin.hh ../XrdCrypto/XrdCryptoFactory.hh \
  ../XrdCrypto/XrdCryptoAux.hh ../XrdCrypto/XrdCryptoMsgDigest.hh \
  ../XrdCrypto/XrdCryptoBasic.hh ../XrdSut/XrdSutBucket.hh \
  ../XrdSut/XrdSutAux.hh ../XrdClient/XrdClientAbsMonIntf.hh \
  ../XrdClient/XrdcpXtremeRead.hh ../XrdClient/XrdClientReadAhead.hh
	@echo Compiling Xrdcp.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFHASXRDCRYPTO) $(INCLUDE) -o $(OBJDIR)/Xrdcp.o Xrdcp.cc

$(OBJDIR)/XrdClientPrep.o: XrdClientPrep.cc ../XProtocol/XProtocol.hh \
                           ../XrdClient/XrdClientAdmin.hh \
                           ../XrdClient/XrdClientEnv.hh \
                           ../XrdSys/XrdSysHeaders.hh
	@echo Compiling XrdClientPrep.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientPrep.o XrdClientPrep.cc

$(OBJDIR)/XrdStageTool.o: XrdStageTool.cc ../XrdClient/XrdClientUrlInfo.hh \
   ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdClient/XrdClient.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdSys/XrdSysHeaders.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdCpMthrQueue.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdSys/XrdSysLogger.hh \
  ../XrdClient/XrdCpWorkLst.hh ../XrdClient/XrdClientAdmin.hh \
  ../XrdClient/XrdClientAbsMonIntf.hh
	@echo Compiling XrdStageTool.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdStageTool.o XrdStageTool.cc

$(OBJDIR)/XrdCommandLine.o: XrdCommandLine.cc ../XrdClient/XrdClientUrlInfo.hh \
  ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdClient/XrdClient.hh \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientUnsolMsg.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientConn.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc ../XrdOuc/XrdOucString.hh \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientDebug.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysLogger.hh \
  ../XrdClient/XrdClientAdmin.hh
	@echo Compiling XrdCommandLine.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFREADLINE) $(INCLUDE) -o $(OBJDIR)/XrdCommandLine.o XrdCommandLine.cc

$(OBJDIR)/XrdClientEnv.o: XrdClientEnv.cc ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientEnv.hh ../XrdOuc/XrdOucEnv.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh ../XrdSys/XrdSysHeaders.hh
	@echo Compiling XrdClientEnv.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientEnv.o XrdClientEnv.cc

$(OBJDIR)/XrdCpWorkLst.o: XrdCpWorkLst.cc ../XrdClient/XrdCpWorkLst.hh \
  ../XrdClient/XrdClientAdmin.hh ../XrdClient/XrdClientAbs.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientConst.hh \
  ../XrdClient/XrdClientReadCache.hh ../XrdClient/XrdClientInputBuffer.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdSys/XrdSysSemWait.hh ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucString.hh \
  ../XrdOuc/XrdOucHash.icc ../XrdClient/XrdClientVector.hh \
  ../XrdSys/XrdSysHeaders.hh \
  ../XrdClient/XrdClient.hh
	@echo Compiling XrdCpWorkLst.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCpWorkLst.o XrdCpWorkLst.cc

$(OBJDIR)/XrdClientSid.o: XrdClientSid.cc ../XrdClient/XrdClientSid.hh \
  ../XrdOuc/XrdOucRash.hh ../XrdOuc/XrdOucRash.icc \
  ../XrdSys/XrdSysPlatform.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientVector.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysHeaders.hh ../XrdSys/XrdSysError.hh
	@echo Compiling XrdClientSid.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientSid.o XrdClientSid.cc

$(OBJDIR)/XrdClientReadV.o: XrdClientReadV.cc ../XrdClient/XrdClientReadV.hh
	@echo Compiling XrdClientReadV.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientReadV.o XrdClientReadV.cc

$(OBJDIR)/XrdClientMStream.o: XrdClientMStream.cc ../XrdClient/XrdClientDebug.hh \
  ../XrdClient/XrdClientConst.hh ../XrdSys/XrdSysPthread.hh \
  ../XrdSys/XrdSysError.hh ../XrdClient/XrdClientEnv.hh \
  ../XrdOuc/XrdOucEnv.hh ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdSys/XrdSysLogger.hh ../XrdClient/XrdClientConnMgr.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientMessage.hh \
  ../XrdClient/XrdClientProtocol.hh ../XProtocol/XProtocol.hh \
  ../XProtocol/XPtypes.hh ../XrdClient/XrdClientSock.hh \
  ../XrdClient/XrdClientUrlInfo.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientLogConnection.hh \
  ../XrdClient/XrdClientPhyConnection.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdClient/XrdClientVector.hh ../XrdClient/XrdClientThread.hh \
  ../XrdClient/XrdClientConn.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdSec/XrdSecInterface.hh ../XrdSec/XrdSecEntity.hh \
  ../XrdNet/XrdNetDNS.hh  \
  ../XrdClient/XrdClientAbs.hh ../XrdClient/XrdClientSid.hh \
  ../XrdOuc/XrdOucRash.hh ../XrdOuc/XrdOucRash.icc \
  ../XrdSys/XrdSysPlatform.hh
	@echo Compiling XrdClientMStream.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdClientMStream.o XrdClientMStream.cc


$(OBJDIR)/XrdcpXtremeRead.o: XrdcpXtremeRead.cc  ../XrdClient/XrdcpXtremeRead.hh \
  ../XrdSys/XrdSysPthread.hh ../XrdSys/XrdSysError.hh \
  ../XrdClient/XrdClient.hh ../XrdClient/XrdClientAbs.hh \
  ../XrdClient/XrdClientUnsolMsg.hh ../XrdClient/XrdClientConn.hh \
  ../XrdClient/XrdClientConst.hh ../XrdClient/XrdClientConnMgr.hh \
  ../XrdOuc/XrdOucHash.hh ../XrdOuc/XrdOucHash.icc \
  ../XrdClient/XrdClientPhyConnection.hh ../XrdClient/XrdClientPSock.hh \
  ../XrdClient/XrdClientSock.hh ../XrdClient/XrdClientUrlInfo.hh \
  ../XrdOuc/XrdOucString.hh ../XrdSys/XrdSysHeaders.hh \
  ../XrdClient/XrdClientVector.hh ../XrdOuc/XrdOucRash.hh \
  ../XrdOuc/XrdOucRash.icc ../XrdSys/XrdSysPlatform.hh \
  ../XrdClient/XrdClientMessage.hh ../XrdClient/XrdClientProtocol.hh \
  ../XProtocol/XProtocol.hh ../XProtocol/XPtypes.hh \
  ../XrdClient/XrdClientInputBuffer.hh ../XrdSys/XrdSysSemWait.hh \
  ../XrdClient/XrdClientThread.hh ../XrdClient/XrdClientReadCache.hh \
  ../XrdVersion.hh ../XrdClient/XrdClientAdmin.hh
	@echo Compiling XrdcpXtremeRead.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdcpXtremeRead.o XrdcpXtremeRead.cc

$(OBJDIR)/XrdClientReadAhead.o: XrdClientReadAhead.cc ../XrdClient/XrdClientReadAhead.hh
	@echo Compiling XrdClientReadAhead.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdClientReadAhead.o XrdClientReadAhead.cc
