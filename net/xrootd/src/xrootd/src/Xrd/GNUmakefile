#              $Id: GNUmakefile,v 1.32 2010/07/30 03:18:29 abh Exp $

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
 
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdBuffer.cc      \
        XrdConfig.cc      \
        XrdInet.cc        \
        XrdInfo.cc        \
        XrdLink.cc        \
        XrdLinkMatch.cc   \
        XrdMain.cc        \
        XrdPoll.cc        \
        XrdProtLoad.cc    \
        XrdProtocol.cc    \
        XrdScheduler.cc   \
        XrdStats.cc

  
OBJECTS = \
        $(OBJDIR)/XrdBuffer.o      \
        $(OBJDIR)/XrdConfig.o      \
        $(OBJDIR)/XrdInet.o        \
        $(OBJDIR)/XrdInfo.o        \
        $(OBJDIR)/XrdLink.o        \
        $(OBJDIR)/XrdLinkMatch.o   \
        $(OBJDIR)/XrdMain.o        \
        $(OBJDIR)/XrdPoll.o        \
        $(OBJDIR)/XrdProtLoad.o    \
        $(OBJDIR)/XrdProtocol.o    \
        $(OBJDIR)/XrdScheduler.o   \
        $(OBJDIR)/XrdStats.o

LIBRARY = $(LIBDIR)/libXrd.a

TARGETS = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOuc% ../XrdOuc
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make Xrd done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJECTS)
	$(ECHO)ranlib $(LIBRARY)

$(OBJDIR)/XrdBuffer.o:    XrdBuffer.cc   XrdBuffer.hh     XrdTrace.hh \
                          XrdSysError.hh XrdSysPthread.hh XrdOucTrace.hh \
                          XrdSysPlatform.hh XrdSysTimer.hh
	@echo Compiling XrdBuffer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBuffer.o XrdBuffer.cc

$(OBJDIR)/XrdConfig.o: XrdBuffer.hh      XrdConfig.hh      XrdJob.hh     \
                       XrdLink.hh        XrdInet.hh        XrdOucUtils.hh \
                       XrdPoll.hh        XrdProtLoad.hh    XrdScheduler.hh \
                       XrdStats.hh       XrdTrace.hh       XrdInfo.hh    \
                       XrdOuca2x.hh      XrdSysError.hh    XrdSysLogger.hh   \
                       XrdOucStream.hh   XrdOucTrace.hh    XrdSysTimer.hh \
                       XrdOucEnv.hh      XrdSysHeaders.hh \
                       XrdNetSecurity.hh XrdNetDNS.hh      XrdConfig.cc \
                       ../XrdVersion.hh
	@echo Compiling XrdConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdConfig.o XrdConfig.cc

$(OBJDIR)/XrdInet.o:    XrdInet.hh         XrdInet.cc   \
                        XrdLink.hh         XrdTrace.hh  \
                        XrdSysError.hh     XrdSysLogger.hh  XrdOucTrace.hh \
                        XrdNet.hh          XrdNetOpts.hh    XrdNetPeer.hh
	@echo Compiling XrdInet.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdInet.o XrdInet.cc

$(OBJDIR)/XrdInfo.o: XrdInfo.hh XrdInfo.cc ../XrdVersion.hh ../../LICENSE
	@echo Compiling XrdInfo.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdInfo.o XrdInfo.cc

$(OBJDIR)/XrdLink.o:   XrdBuffer.hh     XrdJob.hh          XrdLink.hh      \
                       XrdProtocol.hh   XrdScheduler.hh    XrdTrace.hh  \
                       XrdSysError.hh   XrdSysPlatform.hh  XrdSysPthread.hh \
                       XrdOucTrace.hh   XrdLinkMatch.hh    XrdLink.cc \
                       XrdNetDNS.hh     XrdNetPeer.hh
	@echo Compiling XrdLink.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSENDFILE) $(INCLUDE) -o $(OBJDIR)/XrdLink.o XrdLink.cc

$(OBJDIR)/XrdLinkMatch.o: XrdLinkMatch.cc XrdLinkMatch.hh XrdSysPlatform.hh
	@echo Compiling XrdLinkMatch.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdLinkMatch.cc -o $(OBJDIR)/XrdLinkMatch.o

$(OBJDIR)/XrdMain.o:   XrdConfig.hh    XrdLink.hh        XrdInet.hh \
                       XrdPoll.hh      XrdScheduler.hh   XrdTrace.hh    \
                       XrdSysError.hh  XrdSysLogger.hh   XrdSysPthread.hh \
                       XrdOucTrace.hh  XrdProtLoad.hh    XrdSysHeaders.hh \
                       XrdMain.cc
	@echo Compiling XrdMain.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdMain.o XrdMain.cc

$(OBJDIR)/XrdPoll.o:   XrdLink.hh       XrdPoll.hh        XrdProtocol.hh   \
                       XrdPoll.cc       XrdPollDev.hh     XrdPollDev.icc \
                                        XrdPollE.hh       XrdPollE.icc  \
                       XrdPollPoll.hh   XrdPollPoll.icc   XrdProtocol.hh \
                       XrdScheduler.hh  XrdTrace.hh     \
                       XrdSysError.hh   XrdSysPlatform.hh XrdSysPthread.hh \
                       XrdOucTrace.hh
	@echo Compiling XrdPoll.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFDEVPOLL) $(CFEPOLL) $(INCLUDE) -o $(OBJDIR)/XrdPoll.o XrdPoll.cc

$(OBJDIR)/XrdProtLoad.o: XrdJob.hh        XrdLink.hh     XrdPoll.hh \
                         XrdProtLoad.cc   XrdProtLoad.hh \
                         XrdProtocol.hh   XrdTrace.hh \
                         XrdSysError.hh   XrdSysPlugin.hh XrdOucTrace.hh
	@echo Compiling XrdProtLoad.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdProtLoad.o XrdProtLoad.cc

$(OBJDIR)/XrdProtocol.o: XrdJob.hh        XrdProtocol.cc   XrdProtocol.hh
	@echo Compiling XrdProtocol.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdProtocol.o XrdProtocol.cc

$(OBJDIR)/XrdScheduler.o: XrdJob.hh      XrdScheduler.hh   XrdTrace.hh \
                          XrdSysError.hh XrdSysPthread.hh  XrdOucTrace.hh \
                          XrdScheduler.cc
	@echo Compiling XrdScheduler.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdScheduler.o XrdScheduler.cc

$(OBJDIR)/XrdStats.o: XrdBuffer.hh      XrdJob.hh      XrdLink.hh \
                      XrdPoll.hh        XrdProtLoad.hh XrdProtocol.hh \
                      XrdScheduler.hh \
                      XrdStats.cc       XrdStats.hh \
                      XrdNetMsg.hh \
                      XrdSysError.hh    XrdSysPlatform.hh XrdSysPthread.hh \
                      XrdSysTimer.hh    ../XrdVersion.hh
	@echo Compiling XrdStats.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdStats.o XrdStats.cc
