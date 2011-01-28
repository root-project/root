#              $Id: GNUmakefile,v 1.7 2010/07/29 01:24:03 abh Exp $

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

XRDADLERDEPS = $(LIBDIR)/libXrdPosix.a $(LIBDIR)/libXrdClient.a \
               $(LIBDIR)/libXrdOuc.a
XRDADLERLIBS = -L$(LIBDIR) -lXrdPosix -lXrdClient -lXrdOuc \
               $(PTHREAD) $(LIBS) $(LIBZ)
XRDADLER32_OBJ = $(OBJDIR)/Xrdadler32.o
XRDADLER32 = $(BINDIR)/xrdadler32

XrdMPXDEPS = $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdNetUtil.a \
             $(LIBDIR)/libXrdOuc.a
XRDMPXLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys
XRDMPXOBJ  = $(OBJDIR)/XrdMpxStats.o
XRDMPXBIN  = $(BINDIR)/mpxstats

XRDw41DEPS = $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a
XRDW41LIBS = $(LIBS) -L$(LIBDIR) -lXrdOuc -lXrdSys
XRDW41_OBJ = $(OBJDIR)/XrdWait41.o
XRDW41     = $(BINDIR)/wait41

TARGETS = $(XRDADLER32) $(XRDMPXBIN) $(XRDW41)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
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

#------------------------------------------------------------------------------#
#                             x r d a d l e r 3 2                              #
#------------------------------------------------------------------------------#
  
$(XRDADLER32): $(XRDADLER32_OBJ) $(XRDADLERDEPS)
	@echo Creating executable $(XRDADLER32)
	$(ECHO)$(LD) $(LDOP) $(XRDADLER32_OBJ) $(XRDADLERLIBS) -o $(XRDADLER32)

$(XRDADLER32_OBJ): Xrdadler32.cc \
                   XrdPosixExtern.hh    XrdPosixXrootd.hh \
                   XrdClientUrlInfo.hh  XrdClientConst.hh \
                   XrdClient.hh         XrdClientEnv.hh \
                   XrdClientAdmin.hh \
                   XrdOucString.hh
	@echo Compiling Xrdadler32.cc 
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(XRDADLER32_OBJ) Xrdadler32.cc

#------------------------------------------------------------------------------#
#                               m p x s t a t s                                #
#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#
#                                 w a i t 4 1                                  #
#------------------------------------------------------------------------------#
 
$(XRDW41): $(XRDW41_OBJ) $(XRDW41_DEPS)
	@echo Creating executable $(XRDW41)
	$(ECHO)$(LD) $(LDOP) $(XRDW41_OBJ) $(XRDW41LIBS) -o $(XRDW41)

$(XRDW41_OBJ): XrdWait41.cc \
                   XrdOucTList.hh \
                   XrdSysHeaders.hh     XrdSysPlatform.hh \
                   XrdSysPthread.hh
	@echo Compiling XrdWait41.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(XRDW41_OBJ) XrdWait41.cc
