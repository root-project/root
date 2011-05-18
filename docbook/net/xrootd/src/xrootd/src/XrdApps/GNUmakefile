#              $Id$

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

XRDADLERDEPS = $(LIBDIR)/libXrdPosix.a   $(LIBDIR)/libXrdClient.a
XRDADLERDEPT = $(LIBDIR)/libXrdNet.a     $(LIBDIR)/libXrdOuc.a \
               $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdSys.a \
               $(XRDADLERDEPS)
               
XRDADLERLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys \
               $(PTHREAD) $(LIBS) $(LIBZ)
XRDADLER32_OBJ = $(OBJDIR)/Xrdadler32.o
XRDADLER32 = $(BINDIR)/xrdadler32

XrdCCODEPS = $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdNetUtil.a \
             $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a
XRDCCOLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys
XRDCCOOBJ  = $(OBJDIR)/XrdAppsCconfig.o
XRDCCOBIN  = $(BINDIR)/cconfig

XrdMPXDEPS = $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdNetUtil.a \
             $(LIBDIR)/libXrdOuc.a
XRDMPXLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys
XRDMPXOBJ  = $(OBJDIR)/XrdMpxStats.o
XRDMPXBIN  = $(BINDIR)/mpxstats

XRDw41DEPS = $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a
XRDW41LIBS = $(LIBS) -L$(LIBDIR) -lXrdOuc -lXrdSys
XRDW41_OBJ = $(OBJDIR)/XrdWait41.o
XRDW41     = $(BINDIR)/wait41

TARGETS = $(XRDADLER32) $(XRDCCOBIN) $(XRDMPXBIN) $(XRDW41)

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
  
$(XRDADLER32): $(XRDADLER32_OBJ) $(XRDADLERDEPT)
	@echo Creating executable $(XRDADLER32)
	$(ECHO)$(LD) $(LDOP) $(XRDADLER32_OBJ) $(XRDADLERDEPS) $(XRDADLERLIBS) -o $(XRDADLER32)

$(XRDADLER32_OBJ): Xrdadler32.cc \
                   XrdPosixExtern.hh    XrdPosixXrootd.hh \
                   XrdClientUrlInfo.hh  XrdClientConst.hh \
                   XrdClient.hh         XrdClientEnv.hh \
                   XrdClientAdmin.hh \
                   XrdOucString.hh
	@echo Compiling Xrdadler32.cc 
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(XRDADLER32_OBJ) Xrdadler32.cc

#------------------------------------------------------------------------------#
#                              l s c c o n f i g                               #
#------------------------------------------------------------------------------#

$(XRDCCOBIN): $(XRDCCOOBJ) $(XrdCCODEPS)
	@echo Creating executable $(XRDCCOBIN)
	$(ECHO)$(LD) $(LDOP) $(XRDCCOOBJ) $(XRDCCOLIBS) $(LIBS) -o $(XRDCCOBIN)
 
$(XRDCCOOBJ): XrdAppsCconfig.cc XrdNetDNS.hh \
              XrdOucEnv.hh  XrdOucNList.hh XrdOucStream.hh XrdOucUtils.hh \
              XrdSysError.hh XrdSysLogger.hh XrdSysHeaders.hh XrdSysPthread.hh
	@echo Compiling XrdAppsCconfig.cc
	$(ECHO)$(CC) -g -c $(CFLAGS) $(INCLUDE) -o $(XRDCCOOBJ) XrdAppsCconfig.cc

#------------------------------------------------------------------------------#
#                               m p x s t a t s                                #
#------------------------------------------------------------------------------#

$(XRDMPXBIN): $(XRDMPXOBJ) $(XrdMPXDEPS)
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
