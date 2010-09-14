#              $Id: GNUmakefile,v 1.4 2010/07/29 01:24:52 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdCryptoLite -lXrdOuc -lXrdNetUtil -lXrdSys

LIBDEPS = $(LIBDIR)/libXrdCryptoLite.a $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdNetUtil.a    $(LIBDIR)/libXrdSys.a

FSSL = $(SSLEXTRACFLAGS)

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSecProtocolsss.cc \
        XrdSecsssAdmin.cc    \
        XrdSecsssID.cc       \
        XrdSecsssKT.cc

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolsss.o \
        $(OBJDIR)/XrdSecsssAdmin.o    \
        $(OBJDIR)/XrdSecsssID.o       \
        $(OBJDIR)/XrdSecsssKT.o

OBJSBIN = \
        $(OBJDIR)/XrdSecsssAdmin.o \
        $(OBJDIR)/XrdSecsssKT.o

OBJSLIB = \
        $(OBJDIR)/XrdSecProtocolsss.o \
        $(OBJDIR)/XrdSecsssID.o       \
        $(OBJDIR)/XrdSecsssKT.o

TARGBIN = $(BINDIR)/xrdsssadmin

TARGLIB = $(LIBDIR)/libXrdSecsss.$(TYPESHLIB)

TARGETS = $(TARGLIB) $(TARGBIN)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdCrypto% ../XrdCrypto
vpath XrdNet%    ../XrdNet
vpath XrdOuc%    ../XrdOuc
vpath XrdSec%    ../XrdSec
vpath XrdSys%    ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSecsss done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGLIB): $(OBJSLIB) $(LIBDEPS)
	@echo Creating shared library $(TARGLIB)
	$(ECHO)$(CC) $(OBJSLIB) $(LDSO) $(MORELIBS) $(LIBSSL) $(LIBS) -o $(TARGLIB);

$(TARGBIN): $(OBJSBIN)
	@echo Creating executable $(TARGBIN)
	$(ECHO)$(LD) $(LDOP) $(OBJSBIN) $(MORELIBS) $(LIBS) -o $(TARGBIN);

$(OBJDIR)/XrdSecProtocolsss.o: XrdSecProtocolsss.cc    XrdSecProtocolsss.hh \
                               XrdNetDNS.hh            XrdOucErrInfo.hh \
                               XrdOucPup.hh            XrdOucTokenizer.hh \
                               XrdCryptoLite.hh        XrdSecInterface.hh \
                               XrdSecsssID.hh          XrdSecsssKT.hh \
                               XrdSecsssRR.hh  \
                               XrdSysHeaders.hh        XrdSysPlatform.hh \
                               XrdSysPthread.hh
	@echo Compiling XrdSecProtocolsss.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolsss.o XrdSecProtocolsss.cc

$(OBJDIR)/XrdSecsssAdmin.o: XrdSecsssAdmin.cc    XrdSecsssKT.hh \
                            XrdOucErrInfo.hh \
                            XrdSysHeaders.hh     XrdSysPlatform.hh \
                            XrdSysTimer.hh
	@echo Compiling XrdSecsssAdmin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecsssAdmin.o XrdSecsssAdmin.cc

$(OBJDIR)/XrdSecsssID.o: XrdSecsssID.cc    XrdSecsssID.hh \
                         XrdOucHash.hh     XrdOucPup.hh   \
                         XrdSecEntity.hh   XrdSecsssRR.hh \
                         XrdSysHeaders.hh  XrdSysPthread.hh
	@echo Compiling XrdSecsssID.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) $(INCKRB4) -o $(OBJDIR)/XrdSecsssID.o XrdSecsssID.cc

$(OBJDIR)/XrdSecsssKT.o: XrdSecsssKT.cc     XrdSecsssKT.hh \
                         XrdOucErrInfo.hh   XrdOucStream.hh  XrdOucUtils.hh \
                         XrdSysHeaders.hh   XrdSysPthread.hh
	@echo Compiling XrdSecsssKT.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) $(INCKRB4) -o $(OBJDIR)/XrdSecsssKT.o XrdSecsssKT.cc
