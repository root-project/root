#              $Id: GNUmakefile,v 1.17 2010/07/30 03:18:52 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdNet.cc            \
        XrdNetBuffer.cc      \
        XrdNetCmsNotify.cc   \
        XrdNetConnect.cc     \
        XrdNetDNS.cc         \
        XrdNetLink.cc        \
        XrdNetMsg.cc         \
        XrdNetSecurity.cc    \
        XrdNetSocket.cc      \
        XrdNetWork.cc
  
OBJECT1 = \
        $(OBJDIR)/XrdNet.o             \
        $(OBJDIR)/XrdNetBuffer.o       \
        $(OBJDIR)/XrdNetCmsNotify.o    \
        $(OBJDIR)/XrdNetConnect.o      \
        $(OBJDIR)/XrdNetLink.o         \
        $(OBJDIR)/XrdNetMsg.o          \
        $(OBJDIR)/XrdNetSecurity.o     \
        $(OBJDIR)/XrdNetSocket.o       \
        $(OBJDIR)/XrdNetWork.o
  
OBJECT2 = \
        $(OBJDIR)/XrdNetDNS.o

OBJECTS = $(OBJECT1) $(OBJECT2)

LIBNET1 = $(LIBDIR)/libXrdNet.a
LIBNET2 = $(LIBDIR)/libXrdNetUtil.a

TARGETS = $(LIBNET1) $(LIBNET2)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdOuc% ../XrdOuc
vpath XrdNet% ../XrdNet
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdNet done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBNET1): $(OBJECT1)
	@echo Creating archive $(LIBNET1)
	$(ECHO)rm -f $(LIBNET1)
	$(ECHO)ar -rc $(LIBNET1) $(OBJECT1)
	$(ECHO)ranlib $(LIBNET1)

$(LIBNET2): $(OBJECT2)
	@echo Creating archive $(LIBNET2)
	$(ECHO)rm -f $(LIBNET2)
	$(ECHO)ar -rc $(LIBNET2) $(OBJECT2)
	$(ECHO)ranlib $(LIBNET2)

$(OBJDIR)/XrdNet.o: XrdNet.cc       XrdNet.hh         XrdNetBuffer.hh \
                    XrdNetDNS.hh    XrdNetOpts.hh     XrdNetSecurity.hh \
                    XrdNetSocket.hh XrdSysPlatform.hh XrdSysError.hh
	@echo Compiling XrdNet.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNet.cc -o $(OBJDIR)/XrdNet.o

$(OBJDIR)/XrdNetBuffer.o: XrdNetBuffer.cc XrdNetBuffer.hh \
                          XrdOucChain.hh  XrdSysPlatform.hh  XrdSysPthread.hh
	@echo Compiling XrdNetBuffer.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetBuffer.cc -o $(OBJDIR)/XrdNetBuffer.o

$(OBJDIR)/XrdNetCmsNotify.o: XrdNetCmsNotify.cc  XrdNetCmsNotify.hh \
                             XrdNetMsg.hh \
                             XrdOucUtils.hh \
                             XrdSysError.hh      XrdSysPthread.hh \
                             XrdSysTimer.hh
	@echo Compiling XrdNetCmsNotify.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdNetCmsNotify.o XrdNetCmsNotify.cc

$(OBJDIR)/XrdNetConnect.o: XrdNetConnect.cc XrdNetConnect.hh XrdSysPlatform.hh
	@echo Compiling XrdNetConnect.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetConnect.cc -o $(OBJDIR)/XrdNetConnect.o

$(OBJDIR)/XrdNetDNS.o: XrdNetDNS.cc XrdNetDNS.hh XrdSysHeaders.hh \
                       XrdSysPlatform.hh XrdSysPthread.hh
	@echo Compiling XrdNetDNS.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFGETHBYXR) $(CFNAMEINFO) $(CFPROTOR) $(INCLUDE) XrdNetDNS.cc -o $(OBJDIR)/XrdNetDNS.o

$(OBJDIR)/XrdNetLink.o: XrdNetLink.cc XrdNetLink.hh          XrdNet.hh \
                        XrdNetBuffer.hh   XrdNetDNS.hh       XrdNetPeer.hh \
                        XrdOucChain.hh    XrdSysError.hh     XrdSysPthread.hh \
                        XrdSysPlatform.hh XrdOucStream.hh    XrdOucTokenizer.hh
	@echo Compiling XrdNetLink.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetLink.cc -o $(OBJDIR)/XrdNetLink.o

$(OBJDIR)/XrdNetMsg.o: XrdNetMsg.cc   XrdNetMsg.hh  \
                       XrdNet.hh      XrdNetDNS.hh     XrdNetOpts.hh \
                       XrdNetPeer.hh  XrdNetSocket.hh \
                       XrdSysError.hh XrdSysPlatform.hh
	@echo Compiling XrdNetMsg.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetMsg.cc -o $(OBJDIR)/XrdNetMsg.o

$(OBJDIR)/XrdNetSecurity.o: XrdNetSecurity.cc XrdNetSecurity.hh \
                            XrdOucHash.hh     XrdOucHash.icc \
                            XrdOucNList.hh    XrdSysPthread.hh XrdOucTrace.hh
	@echo Compiling XrdNetSecurity.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetSecurity.cc -o $(OBJDIR)/XrdNetSecurity.o

$(OBJDIR)/XrdNetSocket.o: XrdNetSocket.cc  XrdNetSocket.hh \
                          XrdNetConnect.hh XrdNetDNS.hh        XrdNetOpts.hh \
                          XrdSysError.hh   XrdOucUtils.hh XrdSysPlatform.hh
	@echo Compiling XrdNetSocket.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetSocket.cc -o $(OBJDIR)/XrdNetSocket.o

$(OBJDIR)/XrdNetWork.o: XrdNetWork.cc     XrdNetWork.hh     XrdNetBuffer.hh \
                        XrdNetLink.hh     XrdNetOpts.hh     XrdNetPeer.hh   \
                        XrdNet.hh         XrdSysError.hh
	@echo Compiling XrdNetWork.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdNetWork.cc -o $(OBJDIR)/XrdNetWork.o
