#              $Id: GNUmakefile,v 1.5 2009/08/26 22:22:28 abh Exp $

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


XRDw41DEPS = $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a
XRDW41LIBS = $(LIBS) -L$(LIBDIR) -lXrdOuc -lXrdSys

XRDW41_OBJ = $(OBJDIR)/XrdWait41.o
XRDW41     = $(BINDIR)/wait41

TARGETS = $(XRDADLER32) $(XRDMPXBIN) $(XRDW41)

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
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(XRDADLER32_OBJ) Xrdadler32.cc

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

$(XRDW41): $(XRDW41_OBJ) $(XRDW41_DEPS)
	@echo Creating executable $(XRDW41)
	$(ECHO)$(LD) $(LDOP) $(XRDW41_OBJ) $(BINLIBS) $(XRDW41LIBS) -o $(XRDW41)

$(XRDW41_OBJ): XrdWait41.cc \
                   XrdOucTList.hh \
                   XrdSysHeaders.hh     XrdSysPlatform.hh \
                   XrdSysPthread.hh
	@echo Compiling XrdWait41.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(XRDW41_OBJ) XrdWait41.cc
