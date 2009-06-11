#              $Id: GNUmakefile,v 1.2 2009/06/11 11:15:15 ganis Exp $

#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

LIBLIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdSys -lXrdPosix 

BINLIBS = $(LIBDIR)/libXrdPosix.a $(LIBDIR)/libXrdClient.a $(LIBDIR)/libXrdCrypto.a \
          $(LIBDIR)/libXrdSut.a $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a 

XRDADLERLIBS = $(PTHREAD) $(LIBS) $(LIBM) $(LIBZ)

XRDADLER32_OBJ = $(OBJDIR)/Xrdadler32.o
XRDADLER32 = $(BINDIR)/xrdadler32

TARGETS = $(XRDADLER32)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdClient% ../XrdClient
vpath XrdNet%    ../XrdNet
vpath XrdOuc%    ../XrdOuc
vpath XrdPosix%  ../XrdPosix

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
	$(ECHO)$(CC) -g -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(XRDADLER32_OBJ) Xrdadler32.cc
