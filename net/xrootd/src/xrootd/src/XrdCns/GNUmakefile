#              $Id: GNUmakefile,v 1.2 2008/09/07 04:30:55 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdCnsConfig.cc \
        XrdCnsDaemon.cc \
        XrdCnsEvent.cc  \
        XrdCnsMain.cc

OBJECTS  = \
        $(OBJDIR)/XrdCnsConfig.o \
        $(OBJDIR)/XrdCnsDaemon.o \
        $(OBJDIR)/XrdCnsEvent.o  \
        $(OBJDIR)/XrdCnsMain.o

LIBDEP   = $(LIBDIR)/libXrdPosix.a \
           $(LIBDIR)/libXrdClient.a

MORELIBS = -L$(LIBDIR) $(LIBM) -lXrdNet -lXrdOuc -lXrdSys -lXrd

TARGETS  = $(BINDIR)/XrdCnsd

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet%   ../XrdNet
vpath XrdOuc%   ../XrdOuc
vpath XrdPosix% ../XrdPosix
vpath XrdSys%   ../XrdSys
vpath Xrd%      ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdCns done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEP)
	@echo Creating executable $(BINDIR)/XrdCnsd
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(LIBDEP) $(MORELIBS) $(LIBS) -o $(BINDIR)/XrdCnsd

$(OBJDIR)/XrdCnsConfig.o:  XrdCnsDaemon.hh XrdCnsEvent.hh   XrdCnsConfig.cc \
                           XrdNetDNS.hh    XrdNetOpts.hh    XrdNetSocket.hh \
                           XrdScheduler.hh XrdTrace.hh \
                           XrdSysError.hh  XrdSysLogger.hh  XrdOucUtils.hh \
                           XrdSysHeaders.hh XrdSysTimer.hh
	@echo Compiling XrdCnsConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsConfig.o XrdCnsConfig.cc

$(OBJDIR)/XrdCnsDaemon.o: XrdCnsDaemon.hh  XrdCnsDaemon.cc  XrdCnsEvent.hh\
                          XrdTrace.hh      XrdNetDNS.hh  \
                          XrdSysError.hh   XrdOucStream.hh  XrdOucUtils.hh \
                          XrdSysTimer.hh   XrdSysHeaders.hh XrdSysPthread.hh \
                          XrdPosixXrootd.hh
	@echo Compiling XrdCnsDaemon.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsDaemon.o XrdCnsDaemon.cc

$(OBJDIR)/XrdCnsEvent.o: XrdCnsEvent.hh  XrdCnsEvent.cc  \
                         XrdSysError.hh  XrdSysPthread.hh
	@echo Compiling XrdCnsEvent.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsEvent.o XrdCnsEvent.cc

$(OBJDIR)/XrdCnsMain.o:    XrdCnsDaemon.hh    XrdCnsMain.cc \
                           XrdScheduler.hh    XrdTrace.hh \
                           XrdSysError.hh     XrdSysHeaders.hh XrdSysLogger.hh  \
                           XrdSysPlatform.hh  XrdSysPthread.hh
	@echo Compiling XrdCnsMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCnsMain.o XrdCnsMain.cc
