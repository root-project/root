#              $Id: GNUmakefile,v 1.4 2009/07/28 20:01:46 ganis Exp $

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

LIBLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdSys -lXrdPosix 

BINLIBS = $(LIBDIR)/libXrdPosix.a $(LIBDIR)/libXrdClient.a $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a 

XRDMPXLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdSys
XRDMPXOBJ  = $(OBJDIR)/XrdMpxStats.o
XRDMPXBIN  = $(BINDIR)/mpxstats

XRDADLERLIBS = $(PTHREAD) $(LIBS) $(LIBM) $(LIBZ)

XRDADLER32_OBJ = $(OBJDIR)/Xrdadler32.o
XRDADLER32 = $(BINDIR)/xrdadler32

TARGETS = $(XRDADLER32) $(XRDMPXBIN)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdClient% ../XrdClient
vpath XrdNet%    ../XrdNet
vpath XrdOuc%    ../XrdOuc
vpath XrdPosix%  ../XrdPosix
vpath XrdSys%    ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS) 
	@echo Make XrdApps done.

$(XRDADLER32): $(XRDADLER32_OBJ)
	@echo Creating executable $(XRDADLER32)
	$(ECHO)$(LD) $(LDOP) $(XRDADLER32_OBJ) $(BINLIBS) $(XRDADLERLIBS) -o $(XRDADLER32) 

$(XRDADLER32_OBJ): Xrdadler32.cc \
                   XrdPosixExtern.hh    XrdPosixXrootd.hh \
                   XrdClientUrlInfo.hh  XrdClientConst.hh \
                   XrdClient.hh         XrdClientEnv.hh \
                   XrdClientAdmin.hh \
                   XrdOucString.hh
	@echo Compiling Xrdadler32.cc 
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(XRDADLER32_OBJ) Xrdadler32.cc

$(XRDMPXBIN): $(XRDMPXOBJ)
	@echo Creating executable $(XRDMPXBIN)
	$(ECHO)$(LD) $(LDOP) $(XRDMPXOBJ) $(XRDMPXLIBS) $(LIBS) -o $(XRDMPXBIN)

$(XRDMPXOBJ): XrdMpxStats.cc \
              XrdNetDNS.hh       XrdNetSocket.hh   XrdNetOpts.hh \
              XrdOucTokenizer.hh \
              XrdSysError.hh     XrdSysLogger.hh \
              XrdSysHeaders.hh   XrdSysPlatform.hh XrdSysPthread.hh
	@echo Compiling XrdMpxStats.cc 
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(XRDMPXOBJ) XrdMpxStats.cc
